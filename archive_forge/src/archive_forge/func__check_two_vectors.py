import numpy
from numpy import linalg
import warnings
import cupy
from cupy import _core
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device
from cupy.linalg import _util
def _check_two_vectors(x, y):
    if x.ndim != 1:
        raise ValueError('x must be a 1D array (actual: {})'.format(x.ndim))
    if y.ndim != 1:
        raise ValueError('y must be a 1D array (actual: {})'.format(y.ndim))
    if x.size != y.size:
        raise ValueError('x and y must be the same size (actual: {} and {})'.format(x.size, y.size))
    if x.dtype != y.dtype:
        raise TypeError('x and y must be the same dtype (actual: {} and {})'.format(x.dtype, y.dtype))