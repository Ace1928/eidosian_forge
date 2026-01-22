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
def _index_fill(x: TensorLike, dim: int, index: TensorLike, value: Union[NumberType, TensorLike], *, inplace: bool):
    torch._check(index.ndim <= 1, lambda: f'Index should have dimension 1 or 0 (got {index.ndim})')
    if isinstance(value, TensorLike):
        torch._check(value.ndim == 0, lambda: f'Only supports 0-dimensional value tensor. Got a tensor with {value.ndim} dimensions.')
    else:
        value = torch.scalar_tensor(value, dtype=x.dtype, layout=x.layout, device=x.device)
    zero_dim = x.ndim == 0
    y = x.unsqueeze(0) if zero_dim else x
    shape = list(y.shape)
    shape[dim] = index.numel()
    value = value.expand(shape)
    index_copy = Tensor.index_copy_ if inplace else torch.index_copy
    out = index_copy(y, dim, index, value)
    if inplace:
        return x
    else:
        if zero_dim:
            out = out.squeeze(0).clone()
        if out.stride() != x.stride():
            new_out = torch.empty_like(x)
            new_out.copy_(out)
            out = new_out
        return out