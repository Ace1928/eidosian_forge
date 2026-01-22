import numpy
from numpy import linalg
import warnings
import cupy
from cupy import _core
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device
from cupy.linalg import _util
def _iamaxmin(x, out, name):
    if x.ndim != 1:
        raise ValueError('x must be a 1D array (actual: {})'.format(x.ndim))
    dtype = x.dtype.char
    if dtype == 'f':
        t = 's'
    elif dtype == 'd':
        t = 'd'
    elif dtype == 'F':
        t = 'c'
    elif dtype == 'D':
        t = 'z'
    else:
        raise TypeError('invalid dtype')
    func = getattr(cublas, 'i' + t + name)
    handle = device.get_cublas_handle()
    result_dtype = 'i'
    result_ptr, result, orig_mode = _setup_result_ptr(handle, out, result_dtype)
    try:
        func(handle, x.size, x.data.ptr, 1, result_ptr)
    finally:
        cublas.setPointerMode(handle, orig_mode)
    if out is None:
        out = result
    elif out.dtype != result_dtype:
        _core.elementwise_copy(result, out)
    return out