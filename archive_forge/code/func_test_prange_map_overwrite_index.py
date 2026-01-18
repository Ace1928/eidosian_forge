import unittest
from functools import reduce
import numpy as np
from numba import njit, typeof, prange, pndindex
import numba.parfors.parfor
from numba.core import (
from numba.core.registry import cpu_target
from numba.tests.support import (TestCase, is_parfors_unsupported)
def test_prange_map_overwrite_index(self):

    def test_impl():
        n = 20
        arr = np.ones(n)
        for i in prange(n):
            i += 1
            arr[i - 1] = i
        return arr
    with self.assertRaises(errors.UnsupportedRewriteError) as raises:
        self.run_parfor_sub_pass(test_impl, ())
    self.assertIn('Overwrite of parallel loop index', str(raises.exception))