import numpy as np
from numba import vectorize
from numba import cuda, float64
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
def test_vectorize_scalar_arg(self):

    @vectorize(sig, target='cuda')
    def vector_add(a, b):
        return a + b
    A = np.arange(10, dtype=np.float64)
    dA = cuda.to_device(A)
    v = vector_add(1.0, dA)
    np.testing.assert_array_almost_equal(v.copy_to_host(), np.arange(1, 11, dtype=np.float64))