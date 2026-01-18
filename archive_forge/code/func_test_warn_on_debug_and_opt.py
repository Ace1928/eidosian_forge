import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.tests.support import linux_only, override_config
from numba.core.errors import NumbaPerformanceWarning
import warnings
def test_warn_on_debug_and_opt(self):
    with warnings.catch_warnings(record=True) as w:
        cuda.jit(debug=True, opt=True)
    self.assertEqual(len(w), 1)
    self.assertIn('not supported by CUDA', str(w[0].message))