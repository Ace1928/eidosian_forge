from __future__ import annotations
import functools
import inspect
import operator
import typing
import torch
from . import _dtypes, _dtypes_impl, _util
def wrap_tensors(result):
    from ._ndarray import ndarray
    if isinstance(result, torch.Tensor):
        return ndarray(result)
    elif isinstance(result, (tuple, list)):
        result = type(result)((wrap_tensors(x) for x in result))
    return result