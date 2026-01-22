import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim, skip_unless_cudasim
from numba import config, cuda
def _test_array_like_same(self, like_func, array):
    """
        Tests of *_array_like where shape, strides, dtype, and flags should
        all be equal.
        """
    array_like = like_func(array)
    self.assertEqual(array.shape, array_like.shape)
    self.assertEqual(array.strides, array_like.strides)
    self.assertEqual(array.dtype, array_like.dtype)
    self.assertEqual(array.flags['C_CONTIGUOUS'], array_like.flags['C_CONTIGUOUS'])
    self.assertEqual(array.flags['F_CONTIGUOUS'], array_like.flags['F_CONTIGUOUS'])