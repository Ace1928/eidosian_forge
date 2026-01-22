import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate.utils.dataclasses import DistributedType
class RegressionDataset:

    def __init__(self, a=2, b=3, length=64, seed=None):
        rng = np.random.default_rng(seed)
        self.length = length
        self.x = rng.normal(size=(length,)).astype(np.float32)
        self.y = a * self.x + b + rng.normal(scale=0.1, size=(length,)).astype(np.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {'x': self.x[i], 'y': self.y[i]}