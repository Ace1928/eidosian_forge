import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate.utils.dataclasses import DistributedType
class RegressionModel4XPU(torch.nn.Module):

    def __init__(self, a=0, b=0, double_output=False):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([2, 3]).float())
        self.b = torch.nn.Parameter(torch.tensor([2, 3]).float())
        self.first_batch = True

    def forward(self, x=None):
        if self.first_batch:
            print(f'Model dtype: {self.a.dtype}, {self.b.dtype}. Input dtype: {x.dtype}')
            self.first_batch = False
        return x * self.a[0] + self.b[0]