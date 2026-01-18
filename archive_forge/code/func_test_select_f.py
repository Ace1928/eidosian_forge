from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch
def test_select_f(self):
    a = np.arange(5 * 6 * 7).reshape(5, 6, 7, order='F')
    da = cuda.to_device(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            self.assertTrue(np.array_equal(da[i, j, :].copy_to_host(), a[i, j, :]))
        for j in range(a.shape[2]):
            self.assertTrue(np.array_equal(da[i, :, j].copy_to_host(), a[i, :, j]))
    for i in range(a.shape[1]):
        for j in range(a.shape[2]):
            self.assertTrue(np.array_equal(da[:, i, j].copy_to_host(), a[:, i, j]))