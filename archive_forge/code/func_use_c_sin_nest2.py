import sys
import numpy as np
from numba import jit, prange
from numba.core import types
from numba.tests.ctypes_usecases import c_sin
from numba.tests.support import TestCase, captured_stderr
@jit(cache=True, nopython=True)
def use_c_sin_nest2(x):
    return use_c_sin_nest1(x)