import functools
import warnings
import numpy
import cupy
import cupyx.scipy.fft
def _polypow_direct(x, n):
    if n == 0:
        return 1
    if n == 1:
        return x
    if n % 2 == 0:
        return _polypow(cupy.convolve(x, x), n // 2)
    return cupy.convolve(x, _polypow(cupy.convolve(x, x), (n - 1) // 2))