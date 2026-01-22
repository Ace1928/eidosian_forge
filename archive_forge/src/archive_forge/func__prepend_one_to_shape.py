import dataclasses
import typing
from typing import Callable, Optional
import cupy
from cupyx.distributed.array import _array
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _modes
def _prepend_one_to_shape(arr) -> '_array.DistributedArray':
    return _reshape_array_with(arr, lambda shape: (1,) + shape, lambda idx: (slice(None),) + idx)