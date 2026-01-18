import numpy as np
from collections import namedtuple
from numba import void, int32, float32, float64
from numba import guvectorize
from numba import cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
import warnings
from numba.core.errors import NumbaPerformanceWarning
from numba.tests.support import override_config
def test_nopython_flag(self):

    def foo(A, B):
        pass
    guvectorize([void(float32[:], float32[:])], '(x)->(x)', target='cuda', nopython=True)(foo)
    with self.assertRaises(TypeError) as raises:
        guvectorize([void(float32[:], float32[:])], '(x)->(x)', target='cuda', nopython=False)(foo)
    self.assertEqual('nopython flag must be True', str(raises.exception))