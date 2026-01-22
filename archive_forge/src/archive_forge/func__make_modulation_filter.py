from functools import lru_cache
from math import ceil, pi
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn.functional import pad
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import (
def _make_modulation_filter(w0: Tensor, q: int) -> Tensor:
    w0 = torch.tan(w0 / 2)
    b0 = w0 / q
    b = torch.tensor([b0, 0, -b0], dtype=torch.float64)
    a = torch.tensor([1 + b0 + w0 ** 2, 2 * w0 ** 2 - 2, 1 - b0 + w0 ** 2], dtype=torch.float64)
    return torch.stack([b, a], dim=0)