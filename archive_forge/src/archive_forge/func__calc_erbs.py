from functools import lru_cache
from math import ceil, pi
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn.functional import pad
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import (
@lru_cache(maxsize=100)
def _calc_erbs(low_freq: float, fs: int, n_filters: int, device: torch.device) -> Tensor:
    from gammatone.filters import centre_freqs
    ear_q = 9.26449
    min_bw = 24.7
    order = 1
    erbs = ((centre_freqs(fs, n_filters, low_freq) / ear_q) ** order + min_bw ** order) ** (1 / order)
    return torch.tensor(erbs, device=device)