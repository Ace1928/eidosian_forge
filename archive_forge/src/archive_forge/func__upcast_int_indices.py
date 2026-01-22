from __future__ import annotations
import builtins
import math
import operator
from typing import Sequence
import torch
from . import _dtypes, _dtypes_impl, _funcs, _ufuncs, _util
from ._normalizations import (
def _upcast_int_indices(index):
    if isinstance(index, torch.Tensor):
        if index.dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
            return index.to(torch.int64)
    elif isinstance(index, tuple):
        return tuple((_upcast_int_indices(i) for i in index))
    return index