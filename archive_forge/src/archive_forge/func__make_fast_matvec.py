import numpy
import cupy
from cupy import cublas
from cupyx import cusparse
from cupy._core import _dtype
from cupy.cuda import device
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupy_backends.cuda.libs import cublas as _cublas
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse.linalg import _interface
def _make_fast_matvec(A):
    matvec = None
    if _csr.isspmatrix_csr(A) and cusparse.check_availability('spmv'):
        handle = device.get_cusparse_handle()
        op_a = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        alpha = numpy.array(1.0, A.dtype)
        beta = numpy.array(0.0, A.dtype)
        cuda_dtype = _dtype.to_cuda_dtype(A.dtype)
        alg = _cusparse.CUSPARSE_MV_ALG_DEFAULT
        x = cupy.empty((A.shape[0],), dtype=A.dtype)
        y = cupy.empty((A.shape[0],), dtype=A.dtype)
        desc_A = cusparse.SpMatDescriptor.create(A)
        desc_x = cusparse.DnVecDescriptor.create(x)
        desc_y = cusparse.DnVecDescriptor.create(y)
        buff_size = _cusparse.spMV_bufferSize(handle, op_a, alpha.ctypes.data, desc_A.desc, desc_x.desc, beta.ctypes.data, desc_y.desc, cuda_dtype, alg)
        buff = cupy.empty(buff_size, cupy.int8)
        del x, desc_x, y, desc_y

        def matvec(x):
            y = cupy.empty_like(x)
            desc_x = cusparse.DnVecDescriptor.create(x)
            desc_y = cusparse.DnVecDescriptor.create(y)
            _cusparse.spMV(handle, op_a, alpha.ctypes.data, desc_A.desc, desc_x.desc, beta.ctypes.data, desc_y.desc, cuda_dtype, alg, buff.data.ptr)
            return y
    return matvec