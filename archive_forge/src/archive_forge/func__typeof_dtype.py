from collections import namedtuple
from functools import singledispatch
import ctypes
import enum
import numpy as np
from numpy.random.bit_generator import BitGenerator
from numba.core import types, utils, errors
from numba.np import numpy_support
@typeof_impl.register(np.dtype)
def _typeof_dtype(val, c):
    tp = numpy_support.from_dtype(val)
    return types.DType(tp)