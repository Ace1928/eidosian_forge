from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@builtin
def advance(base: tensor, offsets, _builder=None):
    """
    Advance a block pointer

    :param base: the block pointer to advance
    :param offsets: the offsets to advance, a tuple by dimension
    """
    return semantic.advance(base, offsets, _builder)