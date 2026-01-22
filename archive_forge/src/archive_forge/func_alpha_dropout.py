import math
from functools import wraps
from typing import Callable, Optional, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs
from torch._decomp import register_decomposition
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _make_inplace
@register_decomposition(aten.alpha_dropout)
def alpha_dropout(self: TensorLikeType, p: float=0.5, training: bool=False, inplace: bool=False) -> TensorLikeType:
    if inplace:
        raise NotImplementedError
    if not training:
        return self
    torch._check(p <= 1 and p >= 0, lambda: f'dropout probability has to be between 0 and 1, but got, {p}')
    if p == 1:
        return torch.zeros_like(self)
    if p == 0:
        return self
    dropout_mask = _dropout_helper(self, 1 - p)
    alpha = -1.7580993408473766
    a = 1.0 / math.sqrt((alpha * alpha * p + 1) * (1 - p))
    b = torch.logical_not(dropout_mask)
    b = b * (alpha * a) + alpha * a * p
    dropout_mask = a * dropout_mask
    return self * dropout_mask + b