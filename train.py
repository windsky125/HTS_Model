import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import wordnet
from transformers import BertTokenizer, BertForMaskedLM, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

# 加载数据函数
def load_data(json_file):
    """加载 JSON 数据文件并拼接 hts_code 和 description"""
    try:
        data = pd.read_json(json_file)

        # 检查并移除空值
        if data.isnull().values.any():
            print("发现空值，正在移除...")
            data = data.dropna()

        # 拼接 hts_code 和 description
        data["text"] = data.apply(lambda x: f"{x['hts_code']} {x['description']}", axis=1)

        # 将字符串标签转换为数值
        label_mapping = {"匹配": 0, "不匹配": 1, "描述模糊": 2}
        if not set(data["label"].unique()).issubset(label_mapping.keys()):
            raise ValueError("标签中存在未定义的值，请检查输入数据。")

        data["label"] = data["label"].map(label_mapping)
        return data
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

# 数据增强函数
def augment_data(texts, labels):
    """通过同义词替换进行数据增强"""
    augmented_texts, augmented_labels = [], []
    for text, label in zip(texts, labels):
        words = text.split()
        new_words = [
            random.choice(wordnet.synsets(w)[0].lemma_names()) if wordnet.synsets(w) else w
            for w in words
        ]
        augmented_texts.append(" ".join(new_words))
        augmented_labels.append(label)
    return augmented_texts, augmented_labels

# 自定义 Dataset 类
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# 领域预训练
def domain_pretraining(train_texts, model_dir, tokenizer, domain_model_dir, epochs=3, batch_size=8):
    """使用掩码语言模型任务进行领域预训练"""
    print("\n========== 开始领域预训练 ==========")
    model = BertForMaskedLM.from_pretrained(model_dir)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.train()

    # 构建数据集和 DataLoader
    train_encodings = tokenizer(
        train_texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    dataset = torch.utils.data.TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        print(f"\n领域预训练 Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        for batch in tqdm(data_loader):
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # 构建掩码
            labels = input_ids.clone()
            mask = torch.full(input_ids.shape, 0.15)  # 15% mask
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask.bool()] = tokenizer.mask_token_id

            outputs = model(masked_input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(data_loader):.4f}")

    # 保存领域预训练模型
    model.save_pretrained(domain_model_dir)
    tokenizer.save_pretrained(domain_model_dir)
    print("\n领域内预训练完成并保存模型")

# 微调模型
def train_model(train_data, val_data, model_dir, final_model_dir, epochs=5, batch_size=8, learning_rate=2e-5):
    """微调模型"""
    print("\n========== 开始微调 ==========")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=3)  # 假设有3个类别
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 创建数据集和 DataLoader
    train_dataset = CustomDataset(
        train_data["text"], train_data["label"], tokenizer
    )
    val_dataset = CustomDataset(
        val_data["text"], val_data["label"], tokenizer
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 定义类别权重和损失函数
    class_weights = torch.tensor([1.0, 2.0, 3.0], device=device)  # 根据类别比例设置
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(epochs):
        print(f"\n微调 Epoch {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"训练损失: {train_loss / len(train_loader):.4f}")

        # 验证阶段
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"验证损失: {val_loss / len(val_loader):.4f}")
        print(f"验证准确率: {accuracy:.4f}")
        print(classification_report(all_labels, all_preds, target_names=["匹配", "不匹配", "描述模糊"], zero_division=0))

    # 保存微调后的模型到 final_model_dir
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print("\n微调完成并保存模型到：", final_model_dir)

# 主函数
def main(json_file):
    data = load_data(json_file)
    if data is None:
        return

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data["text"].tolist(), data["label"].tolist(), test_size=0.2, random_state=42
    )
    train_texts, train_labels = augment_data(train_texts, train_labels)

    base_model_dir = "C:/Users/24227/Desktop/HTS_Model/bert"  # 初始模型路径
    domain_model_dir = "C:/Users/24227/Desktop/HTS_Model/BERT_Model/Domain_Pretrained"
    final_model_dir = "C:/Users/24227/Desktop/HTS_Model/BERT_Model/Fine_Tuned"

    tokenizer = BertTokenizer.from_pretrained(base_model_dir)

    # 领域预训练
    domain_pretraining(train_texts, base_model_dir, tokenizer, domain_model_dir)

    # 微调
    train_data = {"text": train_texts, "label": train_labels}
    val_data = {"text": val_texts, "label": val_labels}
    train_model(train_data, val_data, domain_model_dir, final_model_dir)  # 传递 final_model_dir

if __name__ == "__main__":
    json_file = "C:/Users/24227/Desktop/1.json"  # 输入的数据文件路径
    main(json_file)
