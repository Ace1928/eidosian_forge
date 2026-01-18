import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def test_set_c(self):
    self._test_set_equal(set_c, 43j, types.complex64)
    self._test_set_equal(set_c, 43j, types.complex128)