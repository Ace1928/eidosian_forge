import torch
from torch import Tensor
import inspect
import warnings
from typing import Dict, List, Optional, Set
from torch.types import Number
@register_decomposition(aten.var.correction)
def var_decomposition(input: Tensor, dim: Optional[List[int]]=None, correction: Optional[Number]=None, keepdim: bool=False) -> Tensor:
    if dim is None:
        dim_i: List[int] = []
        dim = dim_i
    if isinstance(dim, (tuple, list)) and len(dim) == 0:
        n = input.numel()
    else:
        n = 1
        for dim_i in dim:
            n *= input.shape[dim_i]
    mean = aten.mean(input, dim, True)
    sub = input - mean
    sq = sub * sub
    sum = aten.sum(sq, dim, keepdim)
    if correction is None:
        denom = float(n - 1)
    elif isinstance(correction, int):
        denom = float(n - correction)
    elif isinstance(correction, float):
        denom = float(n) - correction
    else:
        raise RuntimeError('correction must be int or float')
    return sum / max(0, denom)