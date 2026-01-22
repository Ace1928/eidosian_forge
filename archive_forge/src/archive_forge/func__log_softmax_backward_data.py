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
@register_decomposition(aten._log_softmax_backward_data)
@out_wrapper()
@compute_only_pw_cast_for_opmath
def _log_softmax_backward_data(grad_output: Tensor, output: Tensor, dim: int, input_dtype: torch.dtype):
    grad_input = grad_output - torch.exp(output) * torch.sum(grad_output, dim=dim, keepdim=True)
    return _cast_grad_to_input_dtype(grad_output, grad_input, input_dtype)