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
@register_meta(torch.ops.aten.transpose_)
def transpose_(self, dim0, dim1):
    assert self.layout not in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}, f'torch.transpose_: in-place transposition is not supported for {self.layout} layout'
    ndims = self.ndim
    dim0 = maybe_wrap_dim(dim0, ndims)
    dim1 = maybe_wrap_dim(dim1, ndims)
    if dim0 == dim1:
        return self
    size = list(self.size())
    stride = list(self.stride())
    stride[dim0], stride[dim1] = (stride[dim1], stride[dim0])
    size[dim0], size[dim1] = (size[dim1], size[dim0])
    self.as_strided_(size, stride)
    return self