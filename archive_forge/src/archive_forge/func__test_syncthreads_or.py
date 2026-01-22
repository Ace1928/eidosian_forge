import numpy as np
from numba import cuda, int32, float32
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
def _test_syncthreads_or(self, in_dtype):
    compiled = cuda.jit(use_syncthreads_or)
    nelem = 100
    ary_in = np.zeros(nelem, dtype=in_dtype)
    ary_out = np.zeros(nelem, dtype=np.int32)
    compiled[1, nelem](ary_in, ary_out)
    self.assertTrue(np.all(ary_out == 0))
    ary_in[31] = 1
    compiled[1, nelem](ary_in, ary_out)
    self.assertTrue(np.all(ary_out == 1))