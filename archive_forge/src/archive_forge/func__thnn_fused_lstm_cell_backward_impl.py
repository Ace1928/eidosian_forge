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
@register_meta(aten._thnn_fused_lstm_cell_backward_impl.default)
def _thnn_fused_lstm_cell_backward_impl(grad_hy, grad_cy, cx, cy, workspace, has_bias):
    if grad_hy is None and grad_cy is None:
        return (None, None, None)
    checkLSTMBackwardSizes(grad_hy, grad_cy, cx, cy, workspace)
    grad_gates = torch.empty_like(workspace, memory_format=legacy_contiguous_memory_format)
    grad_cx = torch.empty_like(cx, memory_format=legacy_contiguous_memory_format)
    grad_bias = grad_gates.sum(0, keepdim=False) if has_bias else None
    return (grad_gates, grad_cx, grad_bias)