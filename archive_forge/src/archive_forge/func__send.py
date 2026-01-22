import numpy
import warnings
import cupy
from cupy.cuda import nccl
from cupyx.distributed import _store
from cupyx.distributed._comm import _Backend
from cupyx.scipy import sparse
@classmethod
def _send(cls, comm, array, peer, dtype, count, stream=None):
    dtype = array.dtype.char
    if dtype not in _nccl_dtypes:
        raise TypeError(f'Unknown dtype {array.dtype} for NCCL')
    dtype, count = comm._get_nccl_dtype_and_count(array)
    stream = comm._get_stream(stream)
    comm._comm.send(array.data.ptr, count, dtype, peer, stream)