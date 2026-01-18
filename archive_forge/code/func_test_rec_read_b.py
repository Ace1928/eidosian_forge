import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def test_rec_read_b(self):
    self._test_rec_read(np.int32(2), record_read_b, 'b')