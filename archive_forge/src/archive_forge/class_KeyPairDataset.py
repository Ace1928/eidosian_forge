import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from ..utils.generic import ModelOutput
class KeyPairDataset(Dataset):

    def __init__(self, dataset: Dataset, key1: str, key2: str):
        self.dataset = dataset
        self.key1 = key1
        self.key2 = key2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return {'text': self.dataset[i][self.key1], 'text_pair': self.dataset[i][self.key2]}