import numpy as np
from numba import cuda, int32, float32
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
def test_simple_smem(self):
    compiled = cuda.jit('void(int32[::1])')(simple_smem)
    nelem = 100
    ary = np.empty(nelem, dtype=np.int32)
    compiled[1, nelem](ary)
    self.assertTrue(np.all(ary == np.arange(nelem, dtype=np.int32)))