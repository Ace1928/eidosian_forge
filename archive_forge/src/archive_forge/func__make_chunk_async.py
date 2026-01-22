from itertools import chain
from typing import Any, Callable, Iterable, Optional
import numpy
from numpy.typing import ArrayLike
from numpy.typing import DTypeLike
import cupy
from cupy._core.core import ndarray
import cupy._creation.from_data as _creation_from_data
import cupy._core._routines_math as _math
import cupy._core._routines_statistics as _statistics
from cupy.cuda.device import Device
from cupy.cuda.stream import Stream
from cupy.cuda.stream import get_current_stream
from cupyx.distributed.array import _chunk
from cupyx.distributed.array._chunk import _Chunk
from cupyx.distributed.array import _data_transfer
from cupyx.distributed.array._data_transfer import _Communicator
from cupyx.distributed.array import _elementwise
from cupyx.distributed.array import _index_arith
from cupyx.distributed.array import _modes
from cupyx.distributed.array import _reduction
from cupyx.distributed.array import _linalg
def _make_chunk_async(src_dev, dst_dev, idx, src_array, comms):
    src_stream = get_current_stream(src_dev)
    with src_array.device:
        src_array = _creation_from_data.ascontiguousarray(src_array)
        src_data = _data_transfer._AsyncData(src_array, src_stream.record(), prevent_gc=src_array)
    with Device(dst_dev):
        dst_stream = get_current_stream()
        copied = _data_transfer._transfer(comms[src_dev], src_stream, src_data, comms[dst_dev], dst_stream, dst_dev)
        return _Chunk(copied.array, copied.ready, idx, prevent_gc=src_data)