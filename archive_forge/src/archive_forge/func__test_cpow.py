import math
import numpy as np
from numba import cuda, float64, int8, int32, void
from numba.cuda.testing import unittest, CUDATestCase
def _test_cpow(self, dtype, func, rtol=1e-07):
    N = 32
    x = random_complex(N).astype(dtype)
    y = random_complex(N).astype(dtype)
    r = np.zeros_like(x)
    cfunc = cuda.jit(func)
    cfunc[1, N](r, x, y)
    np.testing.assert_allclose(r, x ** y, rtol=rtol)
    x = np.asarray([0j, 1j], dtype=dtype)
    y = np.asarray([0j, 1.0], dtype=dtype)
    r = np.zeros_like(x)
    cfunc[1, 2](r, x, y)
    np.testing.assert_allclose(r, x ** y, rtol=rtol)