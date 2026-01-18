from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@builtin
def umulhi(x, y, _builder=None):
    """
    Returns the most significant 32 bits of the product of x and y.

    :param x: the input tensor
    :type x: int32
    :param y: the input tensor
    :type y: int32
    """
    x = _to_tensor(x, _builder)
    y = _to_tensor(y, _builder)
    return semantic.umulhi(x, y, _builder)