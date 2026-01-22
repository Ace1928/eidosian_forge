from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
def _add_scan_docstr(name: str, return_indices_arg: str=None, tie_break_arg: str=None) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = '\n    Returns the {name} of all elements in the :code:`input` tensor along the provided :code:`axis`\n\n    :param input: the input values\n    :param axis: the dimension along which the scan should be done'
        func.__doc__ = docstr.format(name=name)
        return func
    return _decorator