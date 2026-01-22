from __future__ import annotations
import builtins
import itertools
import operator
from typing import Optional, Sequence
import torch
from . import _dtypes_impl, _util
from ._normalizations import (
from torch import broadcast_shapes
def _sort_helper(tensor, axis, kind, order):
    if tensor.dtype.is_complex:
        raise NotImplementedError(f'sorting {tensor.dtype} is not supported')
    (tensor,), axis = _util.axis_none_flatten(tensor, axis=axis)
    axis = _util.normalize_axis_index(axis, tensor.ndim)
    stable = kind == 'stable'
    return (tensor, axis, stable)