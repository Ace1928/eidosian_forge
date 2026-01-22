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
@register_meta([aten._linalg_slogdet.default, aten._linalg_slogdet.sign])
@out_wrapper('sign', 'logabsdet', 'LU', 'pivots')
def _linalg_slogdet(A: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    squareCheckInputs(A, 'linalg.slogdet')
    checkFloatingOrComplex(A, 'linalg.slogdet', False)
    shape = A.shape
    sign = A.new_empty(shape[:-2])
    logabsdet = A.new_empty(shape[:-2], dtype=toRealValueType(A.dtype))
    LU = torch.empty_strided(size=shape, stride=make_contiguous_strides_for(shape, False), dtype=A.dtype, device=A.device)
    pivots = A.new_empty(shape[:-1], dtype=torch.int32)
    return (sign, logabsdet, LU, pivots)