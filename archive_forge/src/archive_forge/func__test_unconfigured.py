from numba import cuda
from numba.core.errors import TypingError
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
def _test_unconfigured(self, kernfunc):
    with self.assertRaises(ValueError) as raises:
        kernfunc(0)
    self.assertIn('launch configuration was not specified', str(raises.exception))