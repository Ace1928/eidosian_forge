import unittest
from functools import reduce
import numpy as np
from numba import njit, typeof, prange, pndindex
import numba.parfors.parfor
from numba.core import (
from numba.core.registry import cpu_target
from numba.tests.support import (TestCase, is_parfors_unsupported)
def test_numpy_arrayexpr_reshaped(self):

    def test_impl(a, b):
        a = a.reshape(1, a.size)
        return a + b
    a = np.ones(10)
    b = np.ones(10)
    args = (a, b)
    argtypes = [typeof(x) for x in args]
    sub_pass = self.run_parfor_sub_pass(test_impl, argtypes)
    self.assertEqual(len(sub_pass.rewritten), 1)
    [record] = sub_pass.rewritten
    self.assertEqual(record['reason'], 'arrayexpr')
    self.check_records(sub_pass.rewritten)
    self.run_parallel(test_impl, *args)