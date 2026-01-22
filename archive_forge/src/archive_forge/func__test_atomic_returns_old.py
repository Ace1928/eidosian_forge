import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def _test_atomic_returns_old(self, kernel, initial):
    x = np.zeros(2, dtype=np.float32)
    x[0] = initial
    kernel[1, 1](x)
    if np.isnan(initial):
        self.assertTrue(np.isnan(x[1]))
    else:
        self.assertEqual(x[1], initial)