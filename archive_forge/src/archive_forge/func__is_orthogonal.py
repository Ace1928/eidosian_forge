from enum import Enum, auto
import torch
from torch import Tensor
from ..utils import parametrize
from ..modules import Module
from .. import functional as F
from typing import Optional
def _is_orthogonal(Q, eps=None):
    n, k = (Q.size(-2), Q.size(-1))
    Id = torch.eye(k, dtype=Q.dtype, device=Q.device)
    eps = 10.0 * n * torch.finfo(Q.dtype).eps
    return torch.allclose(Q.mH @ Q, Id, atol=eps)