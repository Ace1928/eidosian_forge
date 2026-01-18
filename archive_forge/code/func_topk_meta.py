import math
from enum import Enum
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._prims_common as utils
from torch import SymBool, SymFloat, Tensor
from torch._decomp import (
from torch._ops import OpOverload
from torch._prims import _prim_elementwise_meta, ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _broadcast_shapes, _maybe_broadcast
from torch.utils import _pytree as pytree
import torch._refs
import torch._refs.nn.functional
import torch._refs.special
@register_meta(aten.topk.default)
def topk_meta(self, k, dim=-1, largest=True, sorted=True):
    dim = maybe_wrap_dim(dim, self.dim(), wrap_scalar=True)
    torch._check(k >= 0 and k <= (self.size(dim) if self.dim() > 0 else 1), lambda: 'selected index k out of range')
    sliceSize = 1 if self.dim() == 0 else self.size(dim)
    torch._check(k >= 0 and k <= sliceSize, lambda: 'k not in range for dimension')
    topKSize = list(self.shape)
    if len(topKSize) > 0:
        topKSize[dim] = k
    return (self.new_empty(topKSize), self.new_empty(topKSize, dtype=torch.int64))