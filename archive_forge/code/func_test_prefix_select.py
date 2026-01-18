from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch
def test_prefix_select(self):
    arr = np.arange(5 * 7).reshape(5, 7, order='F')
    darr = cuda.to_device(arr)
    self.assertTrue(np.all(darr[:1, 1].copy_to_host() == arr[:1, 1]))