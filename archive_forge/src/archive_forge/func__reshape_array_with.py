import dataclasses
import typing
from typing import Callable, Optional
import cupy
from cupyx.distributed.array import _array
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _modes
def _reshape_array_with(arr: '_array.DistributedArray', f_shape: Callable[[tuple[int, ...]], tuple[int, ...]], f_idx: Callable[[tuple[slice, ...]], tuple[slice, ...]]) -> '_array.DistributedArray':

    def reshape_chunk(chunk: _chunk._Chunk) -> _chunk._Chunk:
        data = chunk.array.reshape(f_shape(chunk.array.shape))
        index = f_idx(chunk.index)
        updates = [(data, f_idx(idx)) for data, idx in chunk.updates]
        return _chunk._Chunk(data, chunk.ready, index, updates, chunk.prevent_gc)
    chunks_map = {}
    for dev, chunks in arr._chunks_map.items():
        chunks_map[dev] = [reshape_chunk(chunk) for chunk in chunks]
    shape = f_shape(arr.shape)
    return _array.DistributedArray(shape, arr.dtype, chunks_map, arr._mode, arr._comms)