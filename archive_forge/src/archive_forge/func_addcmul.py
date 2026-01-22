import builtins
import collections
import inspect
import itertools
import math
import operator
import warnings
from collections.abc import Iterable
from enum import Enum
from functools import partial, reduce, singledispatch, wraps
from typing import Any, Callable, Dict, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch import sym_float, sym_int
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._decomp import register_decomposition
import torch._refs._conversions
import torch._refs.fft
import torch._refs.linalg
import torch._refs.nn.functional
import torch._refs.special
@register_decomposition(aten.addcmul)
@out_wrapper()
@elementwise_type_promotion_wrapper(type_promoting_args=('self', 'tensor1', 'tensor2'), type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def addcmul(self: TensorLikeType, tensor1: TensorLikeType, tensor2: TensorLikeType, *, value: NumberType=1) -> TensorLikeType:
    """
    Reference implementation of torch.addcmul
    """
    if value is not None:
        dtype = self.dtype
        python_type = utils.dtype_to_type(dtype)
        torch._check_value(utils.is_weakly_lesser_type(type(value), python_type), lambda: f'value argument of type {type(value)} cannot be safely cast to type {python_type}!')
    return self + value * tensor1 * tensor2