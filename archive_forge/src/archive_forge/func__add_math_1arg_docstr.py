from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
def _add_math_1arg_docstr(name: str) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = '\n    Computes the element-wise {name} of :code:`x`.\n\n    :param x: the input values\n    :type x: Block\n    '
        func.__doc__ = docstr.format(name=name)
        return func
    return _decorator