from warnings import warn
import numpy
import cupy
from cupy.cuda import cublas
from cupy.cuda import cusolver
from cupy.cuda import device
from cupy.cuda import runtime
from cupy.linalg import _util
from cupyx.scipy.linalg import _uarray
def _lu_factor(a, overwrite_a=False, check_finite=True):
    a = cupy.asarray(a)
    _util._assert_2d(a)
    dtype = a.dtype
    if dtype.char == 'f':
        getrf = cusolver.sgetrf
        getrf_bufferSize = cusolver.sgetrf_bufferSize
    elif dtype.char == 'd':
        getrf = cusolver.dgetrf
        getrf_bufferSize = cusolver.dgetrf_bufferSize
    elif dtype.char == 'F':
        getrf = cusolver.cgetrf
        getrf_bufferSize = cusolver.cgetrf_bufferSize
    elif dtype.char == 'D':
        getrf = cusolver.zgetrf
        getrf_bufferSize = cusolver.zgetrf_bufferSize
    else:
        msg = 'Only float32, float64, complex64 and complex128 are supported.'
        raise NotImplementedError(msg)
    a = a.astype(dtype, order='F', copy=not overwrite_a)
    if check_finite:
        if a.dtype.kind == 'f' and (not cupy.isfinite(a).all()):
            raise ValueError('array must not contain infs or NaNs')
    cusolver_handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.int32)
    m, n = a.shape
    ipiv = cupy.empty((min(m, n),), dtype=numpy.intc)
    buffersize = getrf_bufferSize(cusolver_handle, m, n, a.data.ptr, m)
    workspace = cupy.empty(buffersize, dtype=dtype)
    getrf(cusolver_handle, m, n, a.data.ptr, m, workspace.data.ptr, ipiv.data.ptr, dev_info.data.ptr)
    if not runtime.is_hip and dev_info[0] < 0:
        raise ValueError('illegal value in %d-th argument of internal getrf (lu_factor)' % -dev_info[0])
    elif dev_info[0] > 0:
        warn('Diagonal number %d is exactly zero. Singular matrix.' % dev_info[0], RuntimeWarning, stacklevel=2)
    ipiv -= 1
    return (a, ipiv)