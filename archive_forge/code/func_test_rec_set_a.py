import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def test_rec_set_a(self):
    self._test_rec_set(np.float64(1.5), record_set_a, 'a')