import math
import numpy as np
from numba import int32, uint32, float32, float64, jit, vectorize
from numba.tests.support import tag, CheckWarningsMixin
import unittest
class BaseVectorizeUnrecognizedArg(unittest.TestCase, CheckWarningsMixin):
    """
    Test passing an unrecognized argument to the vectorize decorator.
    """

    def _test_target_unrecognized_arg(self, target, with_sig=True):
        a = np.array([2.0], dtype=np.float32)
        b = np.array([3.0], dtype=np.float32)
        sig = [float32(float32, float32)]
        args = with_sig and [sig] or []
        with self.assertRaises(KeyError) as raises:
            f = vectorize(*args, target=target, nonexistent=2)(vector_add)
            f(a, b)
        self.assertIn('Unrecognized options', str(raises.exception))