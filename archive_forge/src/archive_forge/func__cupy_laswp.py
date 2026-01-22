from warnings import warn
import numpy
import cupy
from cupy.cuda import cublas
from cupy.cuda import cusolver
from cupy.cuda import device
from cupy.cuda import runtime
from cupy.linalg import _util
from cupyx.scipy.linalg import _uarray
def _cupy_laswp(A, k1, k2, ipiv, incx):
    m, n = A.shape
    k = ipiv.shape[0]
    assert 0 <= k1 and k1 <= k2 and (k2 < k)
    assert A._c_contiguous or A._f_contiguous
    _kernel_cupy_laswp(m, n, k1, k2, ipiv, incx, A._c_contiguous, A, size=n)