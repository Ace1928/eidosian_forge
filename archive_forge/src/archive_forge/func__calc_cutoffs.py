from functools import lru_cache
from math import ceil, pi
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn.functional import pad
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import (
def _calc_cutoffs(cfs: Tensor, fs: float, q: int) -> Tuple[Tensor, Tensor]:
    w0 = 2 * pi * cfs / fs
    b0 = torch.tan(w0 / 2) / q
    ll = cfs - b0 * fs / (2 * pi)
    rr = cfs + b0 * fs / (2 * pi)
    return (ll, rr)