import dataclasses
import typing
from typing import Callable, Optional
import cupy
from cupyx.distributed.array import _array
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _modes
def _convert_to_tuples(slices: tuple[slice, ...], shape: tuple[int, ...]) -> tuple[_SliceIndices, ...]:
    assert len(slices) == len(shape)
    return tuple((s.indices(l) for s, l in zip(slices, shape)))