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
def _triplet_margin_with_distance_loss(anchor: TensorLikeType, positive: TensorLikeType, negative: TensorLikeType, *, distance_function: Optional[Callable[[TensorLikeType, TensorLikeType], TensorLikeType]]=None, margin: float=1.0, swap: bool=False, reduction: str='mean') -> TensorLikeType:
    _check_reduction_value(reduction)
    a_dim = anchor.ndim
    p_dim = positive.ndim
    n_dim = negative.ndim
    torch._check(a_dim == p_dim and p_dim == n_dim, lambda: f'The anchor, positive, and negative tensors are expected to have the same number of dimensions, but got: anchor {a_dim}D, positive {p_dim}D, and negative {n_dim}D inputs')
    if distance_function is None:
        distance_function = torch.pairwise_distance
    dist_pos = distance_function(anchor, positive)
    dist_neg = distance_function(anchor, negative)
    if swap:
        dist_swap = distance_function(positive, negative)
        dist_neg = torch.minimum(dist_neg, dist_swap)
    loss = torch.clamp_min(margin + dist_pos - dist_neg, 0)
    return _apply_loss_reduction(loss, reduction)