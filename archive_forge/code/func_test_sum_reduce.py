import numpy as np
from numba import cuda
from numba.core.config import ENABLE_CUDASIM
from numba.cuda.testing import CUDATestCase
import unittest
def test_sum_reduce(self):
    if ENABLE_CUDASIM:
        test_sizes = [1, 16]
    else:
        test_sizes = [1, 15, 16, 17, 127, 128, 129, 1023, 1024, 1025, 1536, 1048576, 1049600, 1049728, 34567]
    for n in test_sizes:
        self._sum_reduce(n)