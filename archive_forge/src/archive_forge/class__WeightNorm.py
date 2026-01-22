from enum import Enum, auto
import torch
from torch import Tensor
from ..utils import parametrize
from ..modules import Module
from .. import functional as F
from typing import Optional
class _WeightNorm(Module):

    def __init__(self, dim: Optional[int]=0) -> None:
        super().__init__()
        if dim is None:
            dim = -1
        self.dim = dim

    def forward(self, weight_g, weight_v):
        return torch._weight_norm(weight_v, weight_g, self.dim)

    def right_inverse(self, weight):
        weight_g = torch.norm_except_dim(weight, 2, self.dim)
        weight_v = weight
        return (weight_g, weight_v)