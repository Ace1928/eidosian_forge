import os
import torch
from transformers import PreTrainedTokenizerFast
from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace, Digits
from sentence_transformers import SentenceTransformer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from torch.utils.data import DataLoader, Dataset
from tkinter import *
from tkinter import filedialog, messagebox
import threading

nltk.download("vader_lexicon")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("omw")
nltk.download("universal_tagset")


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


# Load data from a directory
def load_data_from_directory(directory_path):
    texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                texts.append(file.read())
    return texts


# Initialize and train the tokenizer
def initialize_and_train_tokenizer(texts):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [Whitespace(), Digits(individual_digits=True)]
    )
    trainer = trainers.BpeTrainer(
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    return wrapped_tokenizer


# Semantic and sentiment models
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
sentiment_analyzer = SentimentIntensityAnalyzer()


# Extend tokenization with semantic and sentiment analysis
def advanced_tokenize(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    embeddings = semantic_model.encode(tokens, convert_to_tensor=True)
    sentiments = [
        sentiment_analyzer.polarity_scores(token)["compound"] for token in tokens
    ]
    return tokens, embeddings, sentiments


# Batch processing of texts
def batch_tokenize(texts, tokenizer, batch_size=32):
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    all_tokens, all_embeddings, all_sentiments = [], [], []
    for batch in batches:
        tokens_batch = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        )
        embeddings_batch = semantic_model.encode(batch, convert_to_tensor=True)
        sentiments_batch = [
            sentiment_analyzer.polarity_scores(text)["compound"] for text in batch
        ]
        all_tokens.append(tokens_batch)
        all_embeddings.append(embeddings_batch)
        all_sentiments.append(sentiments_batch)
    return all_tokens, all_embeddings, all_sentiments


# Thread to load data and process
def load_data_thread():
    directory_path = filedialog.askdirectory(title="Select Directory with Text Files")
    if directory_path:
        texts = load_data_from_directory(directory_path)
        tokenizer = initialize_and_train_tokenizer(texts)
        tokens, embeddings, sentiments = batch_tokenize(texts, tokenizer)
        for i, text in enumerate(texts):
            print(f"Text {i+1}: {text}")
            print(f"Tokens: {tokens[i]}")
            print(f"Embeddings: {embeddings[i]}")
            print(f"Sentiments: {sentiments[i]}")


if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    threading.Thread(target=load_data_thread).start()
