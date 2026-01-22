from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@builtin
def arange(start, end, _builder=None):
    """
    Returns contiguous values within the left-closed and right-open interval [:code:`start`, :code:`end`).     End - Start must be less than or equal to TRITON_MAX_TENSOR_NUMEL = 131072

    :param start: Start of the interval. Must be a power of two.
    :type start: int32
    :param end: End of the interval. Must be a power of two > start.
    :type end: int32
    """
    start = _constexpr_to_value(start)
    end = _constexpr_to_value(end)
    return semantic.arange(start, end, _builder)