import unittest
from functools import reduce
import numpy as np
from numba import njit, typeof, prange, pndindex
import numba.parfors.parfor
from numba.core import (
from numba.core.registry import cpu_target
from numba.tests.support import (TestCase, is_parfors_unsupported)
def test_prange_map_nested_prange(self):

    def test_impl():
        n = 20
        arr = np.ones((n, n))
        for i in prange(n):
            for j in prange(i):
                arr[i, j] += i + j * n
        return arr
    sub_pass = self.run_parfor_sub_pass(test_impl, ())
    self.assertEqual(len(sub_pass.rewritten), 2)
    self.check_records(sub_pass.rewritten)
    for record in sub_pass.rewritten:
        self.assertEqual(record['reason'], 'loop')
    self.run_parallel(test_impl)