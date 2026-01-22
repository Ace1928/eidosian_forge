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
@register_meta_foreach([aten._foreach_abs_, aten._foreach_acos_, aten._foreach_asin_, aten._foreach_atan_, aten._foreach_ceil_, aten._foreach_cos_, aten._foreach_cosh_, aten._foreach_erf_, aten._foreach_erfc_, aten._foreach_exp_, aten._foreach_expm1_, aten._foreach_frac_, aten._foreach_floor_, aten._foreach_lgamma_, aten._foreach_log_, aten._foreach_log10_, aten._foreach_log1p_, aten._foreach_log2_, aten._foreach_neg_, aten._foreach_reciprocal_, aten._foreach_round_, aten._foreach_sigmoid_, aten._foreach_sign_, aten._foreach_sin_, aten._foreach_sinh_, aten._foreach_sqrt_, aten._foreach_tan_, aten._foreach_tanh_, aten._foreach_trunc_, aten._foreach_zero_, aten._foreach_add_, aten._foreach_sub_, aten._foreach_mul_, aten._foreach_div_, aten._foreach_clamp_min_, aten._foreach_clamp_max_, aten._foreach_lerp_, aten._foreach_copy_])
def _meta_foreach_inplace(*args, _scalar_op=None, **kwargs):
    _meta_foreach_out_of_place(*args, _scalar_op=_scalar_op, **kwargs)
    return