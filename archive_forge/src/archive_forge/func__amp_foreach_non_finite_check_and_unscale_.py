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
@register_meta(aten._amp_foreach_non_finite_check_and_unscale_.default)
def _amp_foreach_non_finite_check_and_unscale_(self, found_inf, inv_scale):
    torch._check(found_inf.numel() == 1, lambda: 'found_inf must be a 1-element tensor.')
    torch._check(inv_scale.numel() == 1, lambda: 'inv_scale must be a 1-element tensor.')
    torch._check(found_inf.dtype.is_floating_point, lambda: 'found_inf must be a float tensor.')
    torch._check(inv_scale.dtype.is_floating_point, lambda: 'inv_scale must be a float tensor.')