import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, xfail_unless_cudasim, CUDATestCase
from numba.core import config
def test_zero_division_error_in_debug(self):

    @cuda.jit(debug=True, opt=False)
    def f(r, x, y):
        r[0] = y[0] / x[0]
        r[1] = y[0]
    r = np.zeros(2)
    x = np.zeros(1)
    y = np.ones(1)
    if config.ENABLE_CUDASIM:
        exc = FloatingPointError
    else:
        exc = ZeroDivisionError
    with self.assertRaises(exc):
        f[1, 1](r, x, y)
    self.assertEqual(r[0], 0, 'Expected result to be left unset')
    self.assertEqual(r[1], 0, 'Expected execution to stop')