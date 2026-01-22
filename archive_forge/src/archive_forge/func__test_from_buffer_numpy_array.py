import array
import numpy as np
from numba import jit, njit
import numba.core.typing.cffi_utils as cffi_support
from numba.core import types, errors
from numba.tests.support import TestCase, skip_unless_cffi
import numba.tests.cffi_usecases as mod
import unittest
def _test_from_buffer_numpy_array(self, pyfunc, dtype):
    x = np.arange(10).astype(dtype)
    y = np.zeros_like(x)
    cfunc = jit(nopython=True)(pyfunc)
    self.check_vector_sin(cfunc, x, y)