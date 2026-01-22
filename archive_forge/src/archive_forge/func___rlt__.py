from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@builtin
def __rlt__(self, other, _builder=None):
    other = _to_tensor(other, _builder)
    return semantic.less_than(other, self, _builder)