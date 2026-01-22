import math
import einops
import pytest
import torch
from torch import nn
import bitsandbytes as bnb
from tests.helpers import id_formatter
class MLP8bit(torch.nn.Module):

    def __init__(self, dim1, dim2, has_fp16_weights=True, memory_efficient_backward=False, threshold=0.0):
        super().__init__()
        self.fc1 = bnb.nn.Linear8bitLt(dim1, dim2, has_fp16_weights=has_fp16_weights, memory_efficient_backward=memory_efficient_backward, threshold=threshold)
        self.fc2 = bnb.nn.Linear8bitLt(dim2, dim1, has_fp16_weights=has_fp16_weights, memory_efficient_backward=memory_efficient_backward, threshold=threshold)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x