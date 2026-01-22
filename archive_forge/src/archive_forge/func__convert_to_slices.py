import dataclasses
import typing
from typing import Callable, Optional
import cupy
from cupyx.distributed.array import _array
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _modes
def _convert_to_slices(tuples: tuple[_SliceIndices, ...]) -> tuple[slice, ...]:
    return tuple((slice(*t) for t in tuples))