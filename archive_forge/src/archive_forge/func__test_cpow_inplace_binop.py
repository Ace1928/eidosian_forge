import math
import numpy as np
from numba import cuda, float64, int8, int32, void
from numba.cuda.testing import unittest, CUDATestCase
def _test_cpow_inplace_binop(self, dtype, rtol=1e-07):
    N = 32
    x = random_complex(N).astype(dtype)
    y = random_complex(N).astype(dtype)
    r = x ** y
    cfunc = cuda.jit(vec_pow_inplace_binop)
    cfunc[1, N](x, y)
    np.testing.assert_allclose(x, r, rtol=rtol)