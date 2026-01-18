import sys
import unittest
import numpy as np
from numba import njit
from numba.core import types
from numba.tests.support import captured_stdout, TestCase
from numba.np import numpy_support
def test_usecase4(self):
    self._test_usecase2to5(usecase4, self.unaligned_dtype)
    self._test_usecase2to5(usecase4, self.aligned_dtype)