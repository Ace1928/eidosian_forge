import unittest
from functools import reduce
import numpy as np
from numba import njit, typeof, prange, pndindex
import numba.parfors.parfor
from numba.core import (
from numba.core.registry import cpu_target
from numba.tests.support import (TestCase, is_parfors_unsupported)
class TestConvertNumpyPass(BaseTest):
    sub_pass_class = numba.parfors.parfor.ConvertNumpyPass

    def check_numpy_allocators(self, fn):

        def test_impl():
            n = 10
            a = fn(n)
            return a
        sub_pass = self.run_parfor_sub_pass(test_impl, ())
        self.assertEqual(len(sub_pass.rewritten), 1)
        [record] = sub_pass.rewritten
        self.assertEqual(record['reason'], 'numpy_allocator')
        self.check_records(sub_pass.rewritten)
        self.run_parallel(test_impl)

    def check_numpy_random(self, fn):

        def test_impl():
            n = 10
            a = fn(n)
            return a
        sub_pass = self.run_parfor_sub_pass(test_impl, ())
        self.assertEqual(len(sub_pass.rewritten), 1)
        [record] = sub_pass.rewritten
        self.assertEqual(record['reason'], 'numpy_allocator')
        self.check_records(sub_pass.rewritten)
        self.run_parallel_check_output_array(test_impl)

    def test_numpy_allocators(self):
        fns = [np.ones, np.zeros]
        for fn in fns:
            with self.subTest(fn.__name__):
                self.check_numpy_allocators(fn)

    def test_numpy_random(self):
        fns = [np.random.random]
        for fn in fns:
            with self.subTest(fn.__name__):
                self.check_numpy_random(fn)

    def test_numpy_arrayexpr(self):

        def test_impl(a, b):
            return a + b
        a = b = np.ones(10)
        args = (a, b)
        argtypes = [typeof(x) for x in args]
        sub_pass = self.run_parfor_sub_pass(test_impl, argtypes)
        self.assertEqual(len(sub_pass.rewritten), 1)
        [record] = sub_pass.rewritten
        self.assertEqual(record['reason'], 'arrayexpr')
        self.check_records(sub_pass.rewritten)
        self.run_parallel(test_impl, *args)

    def test_numpy_arrayexpr_ufunc(self):

        def test_impl(a, b):
            return np.sin(-a) + np.float64(1) / np.sqrt(b)
        a = b = np.ones(10)
        args = (a, b)
        argtypes = [typeof(x) for x in args]
        sub_pass = self.run_parfor_sub_pass(test_impl, argtypes)
        self.assertEqual(len(sub_pass.rewritten), 1)
        [record] = sub_pass.rewritten
        self.assertEqual(record['reason'], 'arrayexpr')
        self.check_records(sub_pass.rewritten)
        self.run_parallel(test_impl, *args)

    def test_numpy_arrayexpr_boardcast(self):

        def test_impl(a, b):
            return a + b + np.array(1)
        a = np.ones(10)
        b = np.ones((3, 10))
        args = (a, b)
        argtypes = [typeof(x) for x in args]
        sub_pass = self.run_parfor_sub_pass(test_impl, argtypes)
        self.assertEqual(len(sub_pass.rewritten), 1)
        [record] = sub_pass.rewritten
        self.assertEqual(record['reason'], 'arrayexpr')
        self.check_records(sub_pass.rewritten)
        self.run_parallel(test_impl, *args)

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