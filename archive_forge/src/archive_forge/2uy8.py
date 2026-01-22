import os
import torch
from kan import KAN
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
from tkinter import *
from tkinter import filedialog, messagebox
import threading
import json


# Define the custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids.squeeze(), inputs.attention_mask.squeeze()


def load_data_from_directory(directory_path):
    texts = []
    for filename in os.listdir(directory_path):
        if (
            filename.endswith(".txt")
            or filename.endswith(".py")
            or filename.endswith(".md")
        ):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                texts.append(file.read())
    return texts


class KANApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KAN Trainer")

        self.model = None
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.dataset = None
        self.dataloader = None

        self.create_widgets()
        self.load_last_model()

    def create_widgets(self):
        Label(
            self.root, text="Kolmogorov-Arnold Network Trainer", font=("Helvetica", 16)
        ).pack(pady=10)

        self.load_button = Button(
            self.root, text="Load Text Data", command=self.load_data
        )
        self.load_button.pack(pady=5)

        self.train_button = Button(
            self.root, text="Train Model", command=self.train_model, state=DISABLED
        )
        self.train_button.pack(pady=5)

        self.save_button = Button(
            self.root, text="Save Model", command=self.save_model, state=DISABLED
        )
        self.save_button.pack(pady=5)

        self.load_model_button = Button(
            self.root, text="Load Model", command=self.load_model
        )
        self.load_model_button.pack(pady=5)

        self.infer_button = Button(
            self.root, text="Generate Text", command=self.generate_text, state=DISABLED
        )
        self.infer_button.pack(pady=5)

        self.prompt_entry = Entry(self.root, width=50)
        self.prompt_entry.pack(pady=5)

        self.output_text = Text(self.root, height=10, width=50)
        self.output_text.pack(pady=10)

    def load_data(self):
        directory_path = filedialog.askdirectory()
        if directory_path:
            texts = load_data_from_directory(directory_path)
            self.dataset = TextDataset(texts, self.tokenizer)
            self.dataloader = DataLoader(self.dataset, batch_size=8, shuffle=True)
            messagebox.showinfo("Info", f"Loaded {len(texts)} text files.")
            self.train_button.config(state=NORMAL)

    def train_model(self):
        def train():
            self.model = KAN(width=[2, 5, 1], grid=5, k=3)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
            self.model.train()

            epochs = 3
            for epoch in range(epochs):
                for batch in self.dataloader:
                    input_ids, attention_mask = batch
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
                messagebox.showinfo("Info", f"Epoch: {epoch + 1} completed.")

            self.save_button.config(state=NORMAL)
            self.infer_button.config(state=NORMAL)

        threading.Thread(target=train).start()

    def save_model(self):
        save_path = filedialog.askdirectory()
        if save_path:
            self.model.save_pretrained(save_path)
            with open("last_model_path.json", "w") as f:
                json.dump({"last_model_path": save_path}, f)
            messagebox.showinfo("Info", "Model saved successfully.")

    def load_model(self):
        model_path = filedialog.askdirectory()
        if model_path:
            self.model = KAN.from_pretrained(model_path)
            messagebox.showinfo("Info", "Model loaded successfully.")
            self.infer_button.config(state=NORMAL)

    def load_last_model(self):
        try:
            with open("last_model_path.json", "r") as f:
                data = json.load(f)
                last_model_path = data.get("last_model_path")
                if last_model_path:
                    self.model = KAN.from_pretrained(last_model_path)
                    messagebox.showinfo("Info", "Last model loaded successfully.")
                    self.infer_button.config(state=NORMAL)
        except FileNotFoundError:
            pass

    def generate_text(self):
        prompt = self.prompt_entry.get()
        if prompt and self.model:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(inputs.input_ids, max_length=50)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.output_text.delete(1.0, END)
            self.output_text.insert(END, generated_text)


if __name__ == "__main__":
    root = Tk()
    app = KANApp(root)
    root.mainloop()
