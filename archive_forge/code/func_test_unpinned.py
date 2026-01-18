import numpy as np
import platform
from numba import cuda
from numba.cuda.testing import unittest, ContextResettingTestCase
def test_unpinned(self):
    A = np.arange(2 * 1024 * 1024)
    self._run_copies(A)