import unittest
from functools import reduce
import numpy as np
from numba import njit, typeof, prange, pndindex
import numba.parfors.parfor
from numba.core import (
from numba.core.registry import cpu_target
from numba.tests.support import (TestCase, is_parfors_unsupported)
def test_pndindex(self):

    def test_impl():
        n = 20
        arr = np.ones((n, n))
        val = 0
        for idx in pndindex(arr.shape):
            val += idx[0] * idx[1]
        return val
    sub_pass = self.run_parfor_sub_pass(test_impl, ())
    self.assertEqual(len(sub_pass.rewritten), 1)
    self.check_records(sub_pass.rewritten)
    [record] = sub_pass.rewritten
    self.assertEqual(record['reason'], 'loop')
    self.run_parallel(test_impl)