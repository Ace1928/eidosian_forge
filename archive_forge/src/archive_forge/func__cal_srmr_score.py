from functools import lru_cache
from math import ceil, pi
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn.functional import pad
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import (
def _cal_srmr_score(bw: Tensor, avg_energy: Tensor, cutoffs: Tensor) -> Tensor:
    """Calculate srmr score."""
    if cutoffs[4] <= bw and cutoffs[5] > bw:
        kstar = 5
    elif cutoffs[5] <= bw and cutoffs[6] > bw:
        kstar = 6
    elif cutoffs[6] <= bw and cutoffs[7] > bw:
        kstar = 7
    elif cutoffs[7] <= bw:
        kstar = 8
    else:
        raise ValueError('Something wrong with the cutoffs compared to bw values.')
    return torch.sum(avg_energy[:, :4]) / torch.sum(avg_energy[:, 4:kstar])