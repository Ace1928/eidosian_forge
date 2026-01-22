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
def _infer_scalar_type(obj):
    if isinstance(obj, FloatLike):
        return torch.get_default_dtype()
    if isinstance(obj, IntLike) and (not isinstance(obj, bool)):
        return torch.int64
    if isinstance(obj, bool):
        return torch.bool
    if isinstance(obj, complex):
        default_dtype = torch.get_default_dtype()
        if default_dtype is torch.float:
            return torch.cfloat
        elif default_dtype is torch.double:
            return torch.cdouble
        else:
            raise RuntimeError('invalid default scalar type for complex')
    if isinstance(obj, torch.Tensor):
        return obj.dtype
    if isinstance(obj, str):
        raise TypeError(f"new(): invalid data type '{type(obj).__name__}'")
    if isinstance(obj, (list, tuple)):
        scalarType = None
        length = len(obj)
        if length == 0:
            return torch.get_default_dtype()
        for i in range(length):
            cur_item = obj[i]
            '\n            if cur_item is obj:\n                raise TypeError("new(): self-referential lists are incompatible")\n            '
            item_scalarType = _infer_scalar_type(cur_item)
            if scalarType is not None:
                scalarType = torch.promote_types(scalarType, item_scalarType)
            else:
                scalarType = item_scalarType
            if scalarType is torch.cdouble:
                return scalarType
        return scalarType
    raise RuntimeError(f'Could not infer dtype of {type(obj).__name__}')