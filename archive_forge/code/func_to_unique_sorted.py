import dataclasses
import typing
from typing import Callable, Optional
import cupy
from cupyx.distributed.array import _array
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _modes
def to_unique_sorted(partitions):
    if len(partitions) == 0:
        raise RuntimeError('Array has no chunk')
    partitions.sort()
    res = [partitions[0]]
    for x, y in zip(partitions, partitions[1:]):
        if x != y:
            res.append(y)
    return res