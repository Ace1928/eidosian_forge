import unittest
from functools import reduce
import numpy as np
from numba import njit, typeof, prange, pndindex
import numba.parfors.parfor
from numba.core import (
from numba.core.registry import cpu_target
from numba.tests.support import (TestCase, is_parfors_unsupported)
def test_prange_three_args(self):

    def test_impl():
        n = 20
        arr = np.ones(n)
        for i in prange(3, n, 2):
            arr[i] += i
        return arr
    with self.assertRaises(errors.UnsupportedRewriteError) as raises:
        self.run_parfor_sub_pass(test_impl, ())
    self.assertIn('Only constant step size of 1 is supported for prange', str(raises.exception))