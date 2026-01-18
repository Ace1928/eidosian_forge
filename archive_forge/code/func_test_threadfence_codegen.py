import numpy as np
from numba import cuda, int32, float32
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
def test_threadfence_codegen(self):
    sig = (int32[:],)
    compiled = cuda.jit(sig)(use_threadfence)
    ary = np.zeros(10, dtype=np.int32)
    compiled[1, 1](ary)
    self.assertEqual(123 + 321, ary[0])
    if not ENABLE_CUDASIM:
        self.assertIn('membar.gl;', compiled.inspect_asm(sig))