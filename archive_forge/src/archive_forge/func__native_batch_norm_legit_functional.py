import functools
import numbers
import operator
import sys
from enum import Enum
from functools import partial, reduce
from itertools import chain, product
from typing import Callable, cast, Iterable, List, Optional, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch.nn.functional as F
from torch import sym_float, sym_int, Tensor
from torch._decomp import register_decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import IntLike, NumberType, TensorLike, TensorSequenceType
from torch._prims_common.wrappers import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
@register_decomposition(aten._native_batch_norm_legit_functional.default)
def _native_batch_norm_legit_functional(input: Tensor, weight: Optional[Tensor], bias: Optional[Tensor], running_mean: Tensor, running_var: Tensor, training: bool, momentum: float, eps: float) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    output, save_mean, save_rstd, new_running_mean, new_running_var = native_batch_norm_helper(input, weight, bias, running_mean, running_var, training, momentum, eps, True)
    assert new_running_mean is not None, 'new_running_mean should not be None'
    assert new_running_var is not None, 'new_running_var should not be None'
    return (output, save_mean, save_rstd, new_running_mean, new_running_var)