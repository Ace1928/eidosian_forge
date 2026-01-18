from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@builtin
def trans(input, _builder=None):
    """
    Returns a transposed tensor.

    :param input: The input tensor.
    :type input:
    """
    return semantic.trans(input, _builder)