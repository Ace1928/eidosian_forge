import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
def test_ravel_1d(self):
    ary = np.arange(60)
    dary = cuda.to_device(ary)
    for order in 'CFA':
        expect = ary.ravel(order=order)
        dflat = dary.ravel(order=order)
        flat = dflat.copy_to_host()
        self.assertTrue(dary is not dflat)
        self.assertEqual(flat.ndim, 1)
        self.assertPreciseEqual(expect, flat)