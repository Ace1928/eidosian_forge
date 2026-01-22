import typing
from typing import Sequence
from itertools import chain
import cupy
import cupy._creation.basic as _creation_basic
from cupy._core.core import ndarray
from cupy.cuda.device import Device
from cupy.cuda.stream import Stream
from cupy.cuda.stream import get_current_stream
from cupyx.distributed.array import _array
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _data_transfer
from cupyx.distributed.array import _index_arith
from cupyx.distributed.array import _modes
def _find_updates(args: Sequence['_array.DistributedArray'], kwargs: dict[str, '_array.DistributedArray'], dev: int, chunk_i: int) -> list['_data_transfer._PartialUpdate']:
    updates: list[_data_transfer._PartialUpdate] = []
    at_most_one_update = True
    for arg in chain(args, kwargs.values()):
        updates_now = arg._chunks_map[dev][chunk_i].updates
        if updates_now:
            if updates:
                at_most_one_update = False
                break
            updates = updates_now
    if at_most_one_update:
        return updates
    for arg in chain(args, kwargs.values()):
        for chunk in chain.from_iterable(arg._chunks_map.values()):
            chunk.flush(arg._mode)
    return []