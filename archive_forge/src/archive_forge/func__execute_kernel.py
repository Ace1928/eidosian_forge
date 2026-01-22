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
def _execute_kernel(kernel, args: Sequence['_array.DistributedArray'], kwargs: dict[str, '_array.DistributedArray']) -> '_array.DistributedArray':
    args = list(args)
    _change_all_to_replica_mode(args, kwargs)
    out_dtype = None
    out_chunks_map: dict[int, list[_chunk._Chunk]] = {}
    for arg in args or kwargs.values():
        index_map = arg.index_map
        break
    for dev, idxs in index_map.items():
        out_chunks_map[dev] = []
        with Device(dev):
            stream = get_current_stream()
            for chunk_i, idx in enumerate(idxs):
                updates = _find_updates(args, kwargs, dev, chunk_i)
                arg_arrays, kwarg_arrays = _prepare_chunks_array(stream, args, kwargs, dev, chunk_i)
                out_chunk = None
                for data in chain(arg_arrays, kwarg_arrays.values()):
                    if isinstance(data, _chunk._ArrayPlaceholder):
                        assert out_chunk is None
                        out_chunk = _chunk._Chunk.create_placeholder(data.shape, data.device, idx)
                if out_chunk is None:
                    out_array = kernel(*arg_arrays, **kwarg_arrays)
                    out_dtype = out_array.dtype
                    out_chunk = _chunk._Chunk(out_array, stream.record(), idx, prevent_gc=(arg_arrays, kwarg_arrays))
                out_chunks_map[dev].append(out_chunk)
                if not updates:
                    continue
                arg_slices = [None] * len(arg_arrays)
                kwarg_slices = {}
                for update, idx in updates:
                    for i, data in enumerate(arg_arrays):
                        if isinstance(data, _chunk._ArrayPlaceholder):
                            arg_slices[i] = update.array
                        else:
                            arg_slices[i] = data[idx]
                    for k, data in kwarg_arrays.items():
                        if isinstance(data, _chunk._ArrayPlaceholder):
                            kwarg_slices[k] = update.array
                        else:
                            kwarg_slices[k] = data[idx]
                    stream.wait_event(update.ready)
                    out_update_array = kernel(*arg_slices, **kwarg_slices)
                    out_dtype = out_update_array.dtype
                    ready = stream.record()
                    out_update = _data_transfer._AsyncData(out_update_array, ready, prevent_gc=(arg_slices, kwarg_slices))
                    out_chunk.add_update(out_update, idx)
    for chunk in chain.from_iterable(out_chunks_map.values()):
        if not isinstance(chunk.array, (ndarray, _chunk._ArrayPlaceholder)):
            raise RuntimeError('Kernels returning other than signle array are not supported')
    shape = comms = None
    for arg in args or kwargs.values():
        shape = arg.shape
        comms = arg._comms
        break
    assert shape is not None
    return _array.DistributedArray(shape, out_dtype, out_chunks_map, _modes.REPLICA, comms)