import math
import numpy as np
from numba import int32, uint32, float32, float64, jit, vectorize
from numba.tests.support import tag, CheckWarningsMixin
import unittest
class BaseVectorizeNopythonArg(unittest.TestCase, CheckWarningsMixin):
    """
    Test passing the nopython argument to the vectorize decorator.
    """

    def _test_target_nopython(self, target, warnings, with_sig=True):
        a = np.array([2.0], dtype=np.float32)
        b = np.array([3.0], dtype=np.float32)
        sig = [float32(float32, float32)]
        args = with_sig and [sig] or []
        with self.check_warnings(warnings):
            f = vectorize(*args, target=target, nopython=True)(vector_add)
            f(a, b)