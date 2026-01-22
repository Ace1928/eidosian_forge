import functools
import warnings
import numpy
import cupy
import cupyx.scipy.fft
def _polyfit_typecast(x):
    if x.dtype.kind == 'c':
        return x.astype(numpy.complex128, copy=False)
    return x.astype(numpy.float64, copy=False)