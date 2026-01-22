import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
class TestNrtNestedGen(TestCase):

    def test_nrt_nested_gen(self):

        def gen0(arr):
            for i in range(arr.size):
                yield arr

        def factory(gen0):

            def gen1(arr):
                out = np.zeros_like(arr)
                for x in gen0(arr):
                    out = out + x
                return (out, arr)
            return gen1
        py_arr = np.arange(10)
        c_arr = py_arr.copy()
        py_res, py_old = factory(gen0)(py_arr)
        c_gen = jit(nopython=True)(factory(jit(nopython=True)(gen0)))
        c_res, c_old = c_gen(c_arr)
        self.assertIsNot(py_arr, c_arr)
        self.assertIs(py_old, py_arr)
        self.assertIs(c_old, c_arr)
        np.testing.assert_equal(py_res, c_res)
        self.assertRefCountEqual(py_res, c_res)

    @unittest.expectedFailure
    def test_nrt_nested_gen_refct(self):

        def gen0(arr):
            yield arr

        def factory(gen0):

            def gen1(arr):
                for out in gen0(arr):
                    return out
            return gen1
        py_arr = np.arange(10)
        c_arr = py_arr.copy()
        py_old = factory(gen0)(py_arr)
        c_gen = jit(nopython=True)(factory(jit(nopython=True)(gen0)))
        c_old = c_gen(c_arr)
        self.assertIsNot(py_arr, c_arr)
        self.assertIs(py_old, py_arr)
        self.assertIs(c_old, c_arr)
        self.assertRefCountEqual(py_old, c_old)

    def test_nrt_nested_nopython_gen(self):
        """
        Test nesting three generators
        """

        def factory(decor=lambda x: x):

            @decor
            def foo(a, n):
                for i in range(n):
                    yield a[i]
                    a[i] += i

            @decor
            def bar(n):
                a = np.arange(n)
                for i in foo(a, n):
                    yield (i * 2)
                for i in range(a.size):
                    yield a[i]

            @decor
            def cat(n):
                for i in bar(n):
                    yield (i + i)
            return cat
        py_gen = factory()
        c_gen = factory(jit(nopython=True))
        py_res = list(py_gen(10))
        c_res = list(c_gen(10))
        self.assertEqual(py_res, c_res)