import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def test_set_b(self):
    self._test_set_equal(set_b, 123, types.int32)
    self._test_set_equal(set_b, 123, types.float64)