import unittest
from functools import reduce
import numpy as np
from numba import njit, typeof, prange, pndindex
import numba.parfors.parfor
from numba.core import (
from numba.core.registry import cpu_target
from numba.tests.support import (TestCase, is_parfors_unsupported)
class TestPreParforPass(BaseTest):

    class sub_pass_class:

        def __init__(self, pass_states):
            pass

        def run(self, blocks):
            pass

    def test_dtype_conversion(self):

        def test_impl(a):
            b = np.ones(20, dtype=a.dtype)
            return b
        arr = np.arange(10)
        args = (arr,)
        argtypes = [typeof(x) for x in args]
        pre_pass = self.run_parfor_pre_pass(test_impl, argtypes)
        self.assertEqual(pre_pass.stats['replaced_func'], 0)
        self.assertEqual(pre_pass.stats['replaced_dtype'], 1)
        self.run_parallel(test_impl, *args)

    def test_sum_replacement(self):

        def test_impl(a):
            return np.sum(a)
        arr = np.arange(10)
        args = (arr,)
        argtypes = [typeof(x) for x in args]
        pre_pass = self.run_parfor_pre_pass(test_impl, argtypes)
        self.assertEqual(pre_pass.stats['replaced_func'], 1)
        self.assertEqual(pre_pass.stats['replaced_dtype'], 0)
        self.run_parallel(test_impl, *args)

    def test_replacement_map(self):

        def test_impl(a):
            return np.sum(a)
        arr = np.arange(10)
        args = (arr,)
        argtypes = [typeof(x) for x in args]
        swap_map = numba.parfors.parfor.swap_functions_map.copy()
        swap_map.pop(('sum', 'numpy'))
        pre_pass = self.run_parfor_pre_pass(test_impl, argtypes, swap_map)
        self.assertEqual(pre_pass.stats['replaced_func'], 0)
        self.assertEqual(pre_pass.stats['replaced_dtype'], 0)
        self.run_parallel(test_impl, *args)