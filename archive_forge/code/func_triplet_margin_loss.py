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
def triplet_margin_loss(anchor: TensorLikeType, positive: TensorLikeType, negative: TensorLikeType, margin: float=1.0, p: float=2, eps: float=1e-06, swap: bool=False, size_average: Optional[bool]=None, reduce: Optional[bool]=None, reduction: str='mean') -> TensorLikeType:
    if size_average is not None or reduce is not None:
        reduction = _get_string_reduction_arg(size_average=size_average, reduce=reduce)
    return _triplet_margin_with_distance_loss(anchor=anchor, positive=positive, negative=negative, distance_function=lambda x, y: torch.pairwise_distance(x, y, p, eps), margin=margin, swap=swap, reduction=reduction)