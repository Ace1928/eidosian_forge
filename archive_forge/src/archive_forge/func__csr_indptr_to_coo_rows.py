import cupy
from cupy import _core
from cupyx.scipy.sparse._base import isspmatrix
from cupyx.scipy.sparse._base import spmatrix
from cupy_backends.cuda.libs import cusparse
from cupy.cuda import device
from cupy.cuda import runtime
import numpy
def _csr_indptr_to_coo_rows(nnz, Bp):
    out_rows = cupy.empty(nnz, dtype=numpy.int32)
    handle = device.get_cusparse_handle()
    if runtime.is_hip and nnz == 0:
        raise ValueError('hipSPARSE currently cannot handle sparse matrices with null ptrs')
    cusparse.xcsr2coo(handle, Bp.data.ptr, nnz, Bp.size - 1, out_rows.data.ptr, cusparse.CUSPARSE_INDEX_BASE_ZERO)
    return out_rows