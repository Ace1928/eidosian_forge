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
def _check_foreach_binop_tensor_lists(self, other):
    torch._check(isinstance(self, List) and isinstance(other, List), lambda: f'The first two arguments of must be List[Tensor], but got {type(self)} and {type(other)}.')
    torch._check(len(self) > 0 and len(self) == len(other), lambda: f'self and other must be non-empty and match in length, but got {len(self)} and {len(other)}.')