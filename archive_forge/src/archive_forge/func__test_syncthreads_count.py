import numpy as np
from numba import cuda, int32, float32
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
def _test_syncthreads_count(self, in_dtype):
    compiled = cuda.jit(use_syncthreads_count)
    ary_in = np.ones(72, dtype=in_dtype)
    ary_out = np.zeros(72, dtype=np.int32)
    ary_in[31] = 0
    ary_in[42] = 0
    compiled[1, 72](ary_in, ary_out)
    self.assertTrue(np.all(ary_out == 70))