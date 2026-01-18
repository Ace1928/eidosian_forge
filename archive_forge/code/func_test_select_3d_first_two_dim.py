from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch
def test_select_3d_first_two_dim(self):
    arr = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    darr = cuda.to_device(arr)
    for i in range(arr.shape[0]):
        expect = arr[i]
        sliced = darr[i]
        self.assertEqual(expect.shape, sliced.shape)
        self.assertEqual(expect.strides, sliced.strides)
        got = sliced.copy_to_host()
        self.assertTrue(np.all(expect == got))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            expect = arr[i, j]
            sliced = darr[i, j]
            self.assertEqual(expect.shape, sliced.shape)
            self.assertEqual(expect.strides, sliced.strides)
            got = sliced.copy_to_host()
            self.assertTrue(np.all(expect == got))