import atexit
import binascii
import functools
import hashlib
import operator
import os
import time
import numpy
import warnings
from numpy.linalg import LinAlgError
import cupy
from cupy import _core
from cupy import cuda
from cupy.cuda import curand
from cupy.cuda import device
from cupy.random import _kernels
from cupy import _util
import cupyx
def _generate_normal(self, func, size, dtype, *args):
    size = _core.get_size(size)
    element_size = _core.internal.prod(size)
    if element_size % 2 == 0:
        out = cupy.empty(size, dtype=dtype)
        func(self._generator, out.data.ptr, out.size, *args)
        return out
    else:
        out = cupy.empty((element_size + 1,), dtype=dtype)
        func(self._generator, out.data.ptr, out.size, *args)
        return out[:element_size].reshape(size)