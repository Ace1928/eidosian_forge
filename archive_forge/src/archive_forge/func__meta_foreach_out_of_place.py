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
@register_meta_foreach([aten._foreach_abs, aten._foreach_acos, aten._foreach_asin, aten._foreach_atan, aten._foreach_ceil, aten._foreach_cos, aten._foreach_cosh, aten._foreach_erf, aten._foreach_erfc, aten._foreach_exp, aten._foreach_expm1, aten._foreach_frac, aten._foreach_floor, aten._foreach_lgamma, aten._foreach_log, aten._foreach_log10, aten._foreach_log1p, aten._foreach_log2, aten._foreach_neg, aten._foreach_reciprocal, aten._foreach_round, aten._foreach_sigmoid, aten._foreach_sign, aten._foreach_sin, aten._foreach_sinh, aten._foreach_sqrt, aten._foreach_tan, aten._foreach_tanh, aten._foreach_trunc, aten._foreach_zero, aten._foreach_add, aten._foreach_sub, aten._foreach_mul, aten._foreach_div, aten._foreach_clamp_min, aten._foreach_clamp_max, aten._foreach_lerp])
def _meta_foreach_out_of_place(*args, _scalar_op=None, **kwargs):
    torch._check(isinstance(args[0], list), lambda: f'The first argument must be List[Tensor], but got {type(args[0])}.')
    nelem = len(args[0])
    torch._check(nelem > 0, lambda: 'Tensor list must have at least one tensor.')
    nlists = 1
    for iarg, arg in enumerate(args[1:]):
        if isinstance(arg, list):
            nlists += 1
            torch._check(len(arg) == nelem, lambda: f'self and argument-{iarg + 2} must match in length, but got {nelem} and {len(arg)}.')
        elif isinstance(arg, Tensor):
            torch._check(arg.dim() == 0 and arg.numel() == 1, lambda: f'scalar tensor expected to be 0 dim but it has {arg.dim()} dimensions and {arg.numel()} elements.')
        else:
            break
    result = []
    for elem in range(nelem):
        each_args = [args[i][elem] for i in range(nlists)]
        result.append(_scalar_op(*each_args, *args[nlists:], **kwargs))
    return result