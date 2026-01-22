import functools
import warnings
import numpy
import cupy
import cupyx.scipy.fft
def _get_coeffs(x):
    if isinstance(x, cupy.poly1d):
        return x._coeffs
    if cupy.isscalar(x):
        return cupy.atleast_1d(x)
    if isinstance(x, cupy.ndarray):
        x = cupy.atleast_1d(x)
        if x.ndim == 1:
            return x
        raise ValueError('Multidimensional inputs are not supported')
    raise TypeError('Unsupported type')