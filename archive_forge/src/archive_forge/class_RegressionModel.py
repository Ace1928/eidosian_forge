import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate.utils.dataclasses import DistributedType
class RegressionModel(torch.nn.Module):

    def __init__(self, a=0, b=0, double_output=False):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(a).float())
        self.b = torch.nn.Parameter(torch.tensor(b).float())
        self.first_batch = True

    def forward(self, x=None):
        if self.first_batch:
            print(f'Model dtype: {self.a.dtype}, {self.b.dtype}. Input dtype: {x.dtype}')
            self.first_batch = False
        return x * self.a + self.b