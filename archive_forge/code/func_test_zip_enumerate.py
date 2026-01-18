from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
import numpy as np
def test_zip_enumerate(self):

    @cuda.jit
    def zipper_enumerator(x, y, error):
        count = 0
        for (i, xv), yv in zip(enumerate(x), y):
            if i != count:
                error[0] = 1
            if xv != x[i]:
                error[0] = 2
            if yv != y[i]:
                error[0] = 3
            count += 1
        if count != len(x):
            error[0] = 4
    self._test_twoarg_function(zipper_enumerator)