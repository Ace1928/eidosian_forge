import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.tests.support import linux_only, override_config
from numba.core.errors import NumbaPerformanceWarning
import warnings
def test_pinned_warn_on_host_array(self):

    @cuda.jit
    def foo(r, x):
        r[0] = x + 1
    N = 10
    ary = cuda.pinned_array(N, dtype=np.float32)
    with override_config('CUDA_WARN_ON_IMPLICIT_COPY', 1):
        with warnings.catch_warnings(record=True) as w:
            foo[1, N](ary, N)
    self.assertEqual(w[0].category, NumbaPerformanceWarning)
    self.assertIn('Host array used in CUDA kernel will incur', str(w[0].message))
    self.assertIn('copy overhead', str(w[0].message))