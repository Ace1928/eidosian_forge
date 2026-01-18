import numpy as np
from numba import cuda, int32, float32
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
def test_syncthreads_and_downcast(self):
    self._test_syncthreads_and(np.int64)