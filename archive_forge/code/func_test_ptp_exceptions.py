from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_ptp_exceptions(self):
    pyfunc = array_ptp_global
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()
    with self.assertTypingError() as e:
        cfunc(np.array((True, True, False)))
    msg = 'Boolean dtype is unsupported (as per NumPy)'
    self.assertIn(msg, str(e.exception))
    with self.assertRaises(ValueError) as e:
        cfunc(np.array([]))
    msg = 'zero-size array reduction not possible'
    self.assertIn(msg, str(e.exception))