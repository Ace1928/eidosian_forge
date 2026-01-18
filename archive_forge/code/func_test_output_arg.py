import numpy as np
from collections import namedtuple
from itertools import product
from numba import vectorize
from numba import cuda, int32, float32, float64
from numba.cuda.cudadrv.driver import CudaAPIError, driver
from numba.cuda.testing import skip_on_cudasim
from numba.cuda.testing import CUDATestCase
import unittest
def test_output_arg(self):

    @vectorize(signatures, target='cuda')
    def vector_add(a, b):
        return a + b
    A = np.arange(10, dtype=np.float32)
    B = np.arange(10, dtype=np.float32)
    expected = A + B
    actual = np.empty_like(A)
    vector_add(A, B, out=actual)
    np.testing.assert_allclose(expected, actual)
    self.assertEqual(expected.dtype, actual.dtype)