import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, xfail_unless_cudasim, CUDATestCase
from numba.core import config
def test_user_raise(self):

    @cuda.jit(debug=True, opt=False)
    def foo(do_raise):
        if do_raise:
            raise ValueError
    foo[1, 1](False)
    with self.assertRaises(ValueError):
        foo[1, 1](True)