import os
import numpy
from numpy import linalg
import cupy
import cupy._util
from cupy import _core
import cupyx
def _tril(x, k=0):
    _tril_kernel(k, x)
    return x