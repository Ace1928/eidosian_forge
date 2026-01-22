import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
class TestArrayIterators(MemoryLeakMixin, TestCase):
    """
    Test array.flat, np.ndenumerate(), etc.
    """

    def setUp(self):
        super(TestArrayIterators, self).setUp()

    def check_array_iter_1d(self, arr):
        pyfunc = array_iter
        cfunc = njit((typeof(arr),))(pyfunc)
        expected = pyfunc(arr)
        self.assertPreciseEqual(cfunc(arr), expected)

    def check_array_iter_items(self, arr):
        pyfunc = array_iter_items
        cfunc = njit((typeof(arr),))(pyfunc)
        expected = pyfunc(arr)
        self.assertPreciseEqual(cfunc(arr), expected)

    def check_array_view_iter(self, arr, index):
        pyfunc = array_view_iter
        cfunc = njit((typeof(arr), typeof(index)))(pyfunc)
        expected = pyfunc(arr, index)
        self.assertPreciseEqual(cfunc(arr, index), expected)

    def check_array_flat(self, arr, arrty=None):
        out = np.zeros(arr.size, dtype=arr.dtype)
        nb_out = out.copy()
        if arrty is None:
            arrty = typeof(arr)
        cfunc = njit((arrty, typeof(out)))(array_flat)
        array_flat(arr, out)
        cfunc(arr, nb_out)
        self.assertPreciseEqual(out, nb_out)

    def check_array_unary(self, arr, arrty, func):
        cfunc = njit((arrty,))(func)
        self.assertPreciseEqual(cfunc(arr), func(arr))

    def check_array_ndenumerate_sum(self, arr, arrty):
        self.check_array_unary(arr, arrty, array_ndenumerate_sum)

    def test_array_iter(self):
        arr = np.arange(6)
        self.check_array_iter_1d(arr)
        self.check_array_iter_items(arr)
        arr = arr[::2]
        self.assertFalse(arr.flags.c_contiguous)
        self.assertFalse(arr.flags.f_contiguous)
        self.check_array_iter_1d(arr)
        self.check_array_iter_items(arr)
        arr = np.bool_([1, 0, 0, 1])
        self.check_array_iter_1d(arr)
        self.check_array_iter_items(arr)
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.check_array_iter_items(arr)
        self.check_array_iter_items(arr.T)

    def test_array_iter_yielded_order(self):

        @jit(nopython=True)
        def foo(arr):
            t = []
            for y1 in arr:
                for y2 in y1:
                    t.append(y2.ravel())
            return t
        arr = np.arange(24).reshape((2, 3, 4), order='F')
        expected = foo.py_func(arr)
        got = foo(arr)
        self.assertPreciseEqual(expected, got)
        arr = np.arange(64).reshape((4, 8, 2), order='F')[::2, :, :]
        expected = foo.py_func(arr)
        got = foo(arr)
        self.assertPreciseEqual(expected, got)
        arr = np.arange(64).reshape((4, 8, 2), order='F')[:, ::2, :]
        expected = foo.py_func(arr)
        got = foo(arr)
        self.assertPreciseEqual(expected, got)
        arr = np.arange(64).reshape((4, 8, 2), order='F')[:, :, ::2]
        expected = foo.py_func(arr)
        got = foo(arr)
        self.assertPreciseEqual(expected, got)

        @jit(nopython=True)
        def flag_check(arr):
            out = []
            for sub in arr:
                out.append((sub, sub.flags.c_contiguous, sub.flags.f_contiguous))
            return out
        arr = np.arange(10).reshape((2, 5), order='F')
        expected = flag_check.py_func(arr)
        got = flag_check(arr)
        self.assertEqual(len(expected), len(got))
        ex_arr, e_flag_c, e_flag_f = expected[0]
        go_arr, g_flag_c, g_flag_f = got[0]
        np.testing.assert_allclose(ex_arr, go_arr)
        self.assertEqual(e_flag_c, g_flag_c)
        self.assertEqual(e_flag_f, g_flag_f)

    def test_array_view_iter(self):
        arr = np.arange(12).reshape((3, 4))
        self.check_array_view_iter(arr, 1)
        self.check_array_view_iter(arr.T, 1)
        arr = arr[::2]
        self.check_array_view_iter(arr, 1)
        arr = np.bool_([1, 0, 0, 1]).reshape((2, 2))
        self.check_array_view_iter(arr, 1)

    def test_array_flat_3d(self):
        arr = np.arange(24).reshape(4, 2, 3)
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 3)
        self.assertEqual(arrty.layout, 'C')
        self.assertTrue(arr.flags.c_contiguous)
        self.check_array_flat(arr)
        arr = arr.transpose()
        self.assertFalse(arr.flags.c_contiguous)
        self.assertTrue(arr.flags.f_contiguous)
        self.assertEqual(typeof(arr).layout, 'F')
        self.check_array_flat(arr)
        arr = arr[::2]
        self.assertFalse(arr.flags.c_contiguous)
        self.assertFalse(arr.flags.f_contiguous)
        self.assertEqual(typeof(arr).layout, 'A')
        self.check_array_flat(arr)
        arr = np.bool_([1, 0, 0, 1] * 2).reshape((2, 2, 2))
        self.check_array_flat(arr)

    def test_array_flat_empty(self):

        def check(arr, arrty):
            cfunc = njit((arrty,))(array_flat_sum)
            cres = cfunc.overloads[arrty,]
            got = cres.entry_point(arr)
            expected = cfunc.py_func(arr)
            self.assertPreciseEqual(expected, got)
        arr = np.zeros(0, dtype=np.int32)
        arr = arr.reshape(0, 2)
        arrty = types.Array(types.int32, 2, layout='C')
        check(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='F')
        check(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='A')
        check(arr, arrty)
        arr = arr.reshape(2, 0)
        arrty = types.Array(types.int32, 2, layout='C')
        check(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='F')
        check(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='A')
        check(arr, arrty)

    def test_array_flat_getitem(self):
        pyfunc = array_flat_getitem
        cfunc = njit(pyfunc)

        def check(arr, ind):
            expected = pyfunc(arr, ind)
            self.assertEqual(cfunc(arr, ind), expected)
        arr = np.arange(24).reshape(4, 2, 3)
        for i in range(arr.size):
            check(arr, i)
        arr = arr.T
        for i in range(arr.size):
            check(arr, i)
        arr = arr[::2]
        for i in range(arr.size):
            check(arr, i)
        arr = np.array([42]).reshape(())
        for i in range(arr.size):
            check(arr, i)
        arr = np.bool_([1, 0, 0, 1])
        for i in range(arr.size):
            check(arr, i)
        arr = arr[::2]
        for i in range(arr.size):
            check(arr, i)

    def test_array_flat_setitem(self):
        pyfunc = array_flat_setitem
        cfunc = njit(pyfunc)

        def check(arr, ind):
            expected = np.copy(arr)
            got = np.copy(arr)
            pyfunc(expected, ind, 123)
            cfunc(got, ind, 123)
            self.assertPreciseEqual(got, expected)
        arr = np.arange(24).reshape(4, 2, 3)
        for i in range(arr.size):
            check(arr, i)
        arr = arr.T
        for i in range(arr.size):
            check(arr, i)
        arr = arr[::2]
        for i in range(arr.size):
            check(arr, i)
        arr = np.array([42]).reshape(())
        for i in range(arr.size):
            check(arr, i)
        arr = np.bool_([1, 0, 0, 1])
        for i in range(arr.size):
            check(arr, i)
        arr = arr[::2]
        for i in range(arr.size):
            check(arr, i)

    def test_array_flat_len(self):
        pyfunc = array_flat_len
        cfunc = njit(array_flat_len)

        def check(arr):
            expected = pyfunc(arr)
            self.assertPreciseEqual(cfunc(arr), expected)
        arr = np.arange(24).reshape(4, 2, 3)
        check(arr)
        arr = arr.T
        check(arr)
        arr = arr[::2]
        check(arr)
        arr = np.array([42]).reshape(())
        check(arr)

    def test_array_flat_premature_free(self):
        cfunc = njit((types.intp,))(array_flat_premature_free)
        expect = array_flat_premature_free(6)
        got = cfunc(6)
        self.assertTrue(got.sum())
        self.assertPreciseEqual(expect, got)

    def test_array_ndenumerate_2d(self):
        arr = np.arange(12).reshape(4, 3)
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 2)
        self.assertEqual(arrty.layout, 'C')
        self.assertTrue(arr.flags.c_contiguous)
        self.check_array_ndenumerate_sum(arr, arrty)
        arr = arr.transpose()
        self.assertFalse(arr.flags.c_contiguous)
        self.assertTrue(arr.flags.f_contiguous)
        arrty = typeof(arr)
        self.assertEqual(arrty.layout, 'F')
        self.check_array_ndenumerate_sum(arr, arrty)
        arr = arr[::2]
        self.assertFalse(arr.flags.c_contiguous)
        self.assertFalse(arr.flags.f_contiguous)
        arrty = typeof(arr)
        self.assertEqual(arrty.layout, 'A')
        self.check_array_ndenumerate_sum(arr, arrty)
        arr = np.bool_([1, 0, 0, 1]).reshape((2, 2))
        self.check_array_ndenumerate_sum(arr, typeof(arr))

    def test_array_ndenumerate_empty(self):

        def check(arr, arrty):
            cfunc = njit((arrty,))(array_ndenumerate_sum)
            cres = cfunc.overloads[arrty,]
            got = cres.entry_point(arr)
            expected = cfunc.py_func(arr)
            np.testing.assert_allclose(expected, got)
        arr = np.zeros(0, dtype=np.int32)
        arr = arr.reshape(0, 2)
        arrty = types.Array(types.int32, 2, layout='C')
        check(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='F')
        check(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='A')
        check(arr, arrty)
        arr = arr.reshape(2, 0)
        arrty = types.Array(types.int32, 2, layout='C')
        check(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='F')
        check(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='A')
        check(arr, arrty)

    def test_array_ndenumerate_premature_free(self):
        cfunc = njit((types.intp,))(array_ndenumerate_premature_free)
        expect = array_ndenumerate_premature_free(6)
        got = cfunc(6)
        self.assertTrue(got.sum())
        self.assertPreciseEqual(expect, got)

    def test_np_ndindex(self):
        func = np_ndindex
        cfunc = njit((types.int32, types.int32))(func)
        self.assertPreciseEqual(cfunc(3, 4), func(3, 4))
        self.assertPreciseEqual(cfunc(3, 0), func(3, 0))
        self.assertPreciseEqual(cfunc(0, 3), func(0, 3))
        self.assertPreciseEqual(cfunc(0, 0), func(0, 0))

    def test_np_ndindex_array(self):
        func = np_ndindex_array
        arr = np.arange(12, dtype=np.int32) + 10
        self.check_array_unary(arr, typeof(arr), func)
        arr = arr.reshape((4, 3))
        self.check_array_unary(arr, typeof(arr), func)
        arr = arr.reshape((2, 2, 3))
        self.check_array_unary(arr, typeof(arr), func)

    def test_np_ndindex_empty(self):
        func = np_ndindex_empty
        cfunc = njit(())(func)
        self.assertPreciseEqual(cfunc(), func())

    def test_iter_next(self):
        func = iter_next
        arr = np.arange(12, dtype=np.int32) + 10
        self.check_array_unary(arr, typeof(arr), func)