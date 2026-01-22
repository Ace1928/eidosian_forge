from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@builtin
def _reduce_with_indices(input, axis, combine_fn, _builder=None, _generator=None):
    axis = _constexpr_to_value(axis)
    n = input.shape[axis]
    index = arange(0, n, _builder=_builder)
    if len(input.shape) > 1:
        axes_to_expand = [constexpr(d) for d in range(len(input.shape))]
        del axes_to_expand[axis]
        index = expand_dims(index, axes_to_expand, _builder=_builder)
        index = broadcast_to(index, input.shape, _builder=_builder)
    rvalue, rindices = reduce((input, index), axis, combine_fn, _builder=_builder, _generator=_generator)
    return (rvalue, rindices)