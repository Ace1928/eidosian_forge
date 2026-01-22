from enum import Enum, auto
import torch
from torch import Tensor
from ..utils import parametrize
from ..modules import Module
from .. import functional as F
from typing import Optional
def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
    assert weight.ndim > 1
    if self.dim != 0:
        weight = weight.permute(self.dim, *(d for d in range(weight.dim()) if d != self.dim))
    return weight.flatten(1)