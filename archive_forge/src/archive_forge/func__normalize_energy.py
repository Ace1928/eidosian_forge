from functools import lru_cache
from math import ceil, pi
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn.functional import pad
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import (
def _normalize_energy(energy: Tensor, drange: float=30.0) -> Tensor:
    """Normalize energy to a dynamic range of 30 dB.

    Args:
        energy: shape [B, N_filters, 8, n_frames]
        drange: dynamic range in dB

    """
    peak_energy = torch.mean(energy, dim=1, keepdim=True).max(dim=2, keepdim=True).values
    peak_energy = peak_energy.max(dim=3, keepdim=True).values
    min_energy = peak_energy * 10.0 ** (-drange / 10.0)
    energy = torch.where(energy < min_energy, min_energy, energy)
    return torch.where(energy > peak_energy, peak_energy, energy)