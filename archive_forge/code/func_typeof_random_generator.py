from collections import namedtuple
from functools import singledispatch
import ctypes
import enum
import numpy as np
from numpy.random.bit_generator import BitGenerator
from numba.core import types, utils, errors
from numba.np import numpy_support
@typeof_impl.register(np.random.Generator)
def typeof_random_generator(val, c):
    return types.NumPyRandomGeneratorType(val)