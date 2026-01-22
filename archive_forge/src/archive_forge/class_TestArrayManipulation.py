from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
class TestArrayManipulation(MemoryLeakMixin, TestCase):
    """
    Check shape-changing operations on arrays.
    """

    def test_array_reshape(self):
        pyfuncs_to_use = [array_reshape, numpy_array_reshape]

        def generic_run(pyfunc, arr, shape):
            return pyfunc(arr, shape)

        @from_generic(pyfuncs_to_use)
        def check(pyfunc, arr, shape):
            expected = pyfunc.py_func(arr, shape)
            self.memory_leak_setup()
            got = generic_run(pyfunc, arr, shape)
            self.assertPreciseEqual(got, expected)
            del got
            self.memory_leak_teardown()

        @from_generic(pyfuncs_to_use)
        def check_only_shape(pyfunc, arr, shape, expected_shape):
            self.memory_leak_setup()
            got = generic_run(pyfunc, arr, shape)
            self.assertEqual(got.shape, expected_shape)
            self.assertEqual(got.size, arr.size)
            del got
            self.memory_leak_teardown()

        @from_generic(pyfuncs_to_use)
        def check_err_shape(pyfunc, arr, shape):
            with self.assertRaises(NotImplementedError) as raises:
                generic_run(pyfunc, arr, shape)
            self.assertEqual(str(raises.exception), 'incompatible shape for array')

        @from_generic(pyfuncs_to_use)
        def check_err_size(pyfunc, arr, shape):
            with self.assertRaises(ValueError) as raises:
                generic_run(pyfunc, arr, shape)
            self.assertEqual(str(raises.exception), 'total size of new array must be unchanged')

        @from_generic(pyfuncs_to_use)
        def check_err_multiple_negative(pyfunc, arr, shape):
            with self.assertRaises(ValueError) as raises:
                generic_run(pyfunc, arr, shape)
            self.assertEqual(str(raises.exception), 'multiple negative shape values')
        arr = np.arange(24)
        check(arr, (24,))
        check(arr, (4, 6))
        check(arr, (8, 3))
        check(arr, (8, 1, 3))
        check(arr, (1, 8, 1, 1, 3, 1))
        arr = np.arange(24).reshape((2, 3, 4))
        check(arr, (24,))
        check(arr, (4, 6))
        check(arr, (8, 3))
        check(arr, (8, 1, 3))
        check(arr, (1, 8, 1, 1, 3, 1))
        check_err_size(arr, ())
        check_err_size(arr, (25,))
        check_err_size(arr, (8, 4))
        arr = np.arange(24).reshape((1, 8, 1, 1, 3, 1))
        check(arr, (24,))
        check(arr, (4, 6))
        check(arr, (8, 3))
        check(arr, (8, 1, 3))
        arr = np.arange(24).reshape((2, 3, 4)).T
        check(arr, (4, 3, 2))
        check(arr, (1, 4, 1, 3, 1, 2, 1))
        check_err_shape(arr, (2, 3, 4))
        check_err_shape(arr, (6, 4))
        check_err_shape(arr, (2, 12))
        arr = np.arange(25).reshape(5, 5)
        check(arr, -1)
        check(arr, (-1,))
        check(arr, (-1, 5))
        check(arr, (5, -1, 5))
        check(arr, (5, 5, -1))
        check_err_size(arr, (-1, 4))
        check_err_multiple_negative(arr, (-1, -2, 5, 5))
        check_err_multiple_negative(arr, (5, 5, -1, -1))

        def check_empty(arr):
            check(arr, 0)
            check(arr, (0,))
            check(arr, (1, 0, 2))
            check(arr, (0, 55, 1, 0, 2))
            check_only_shape(arr, -1, (0,))
            check_only_shape(arr, (-1,), (0,))
            check_only_shape(arr, (0, -1), (0, 0))
            check_only_shape(arr, (4, -1), (4, 0))
            check_only_shape(arr, (-1, 0, 4), (0, 0, 4))
            check_err_size(arr, ())
            check_err_size(arr, 1)
            check_err_size(arr, (1, 2))
        arr = np.array([])
        check_empty(arr)
        check_empty(arr.reshape((3, 2, 0)))
        self.disable_leak_check()

    def test_array_transpose_axes(self):
        pyfuncs_to_use = [numpy_transpose_array_axes_kwarg, numpy_transpose_array_axes_kwarg_copy, array_transpose_axes, array_transpose_axes_copy]

        @from_generic(pyfuncs_to_use)
        def check(pyfunc, arr, axes):
            expected = pyfunc.py_func(arr, axes)
            got = pyfunc(arr, axes)
            self.assertPreciseEqual(got, expected)
            self.assertEqual(got.flags.f_contiguous, expected.flags.f_contiguous)
            self.assertEqual(got.flags.c_contiguous, expected.flags.c_contiguous)

        @from_generic(pyfuncs_to_use)
        def check_err_axis_repeated(pyfunc, arr, axes):
            with self.assertRaises(ValueError) as raises:
                pyfunc(arr, axes)
            self.assertEqual(str(raises.exception), 'repeated axis in transpose')

        @from_generic(pyfuncs_to_use)
        def check_err_axis_oob(pyfunc, arr, axes):
            with self.assertRaises(ValueError) as raises:
                pyfunc(arr, axes)
            self.assertEqual(str(raises.exception), 'axis is out of bounds for array of given dimension')

        @from_generic(pyfuncs_to_use)
        def check_err_invalid_args(pyfunc, arr, axes):
            with self.assertRaises((TypeError, TypingError)):
                pyfunc(arr, axes)
        arrs = [np.arange(24), np.arange(24).reshape(4, 6), np.arange(24).reshape(2, 3, 4), np.arange(24).reshape(1, 2, 3, 4), np.arange(64).reshape(8, 4, 2)[::3, ::2, :]]
        for i in range(len(arrs)):
            check(arrs[i], None)
            for axes in permutations(tuple(range(arrs[i].ndim))):
                ndim = len(axes)
                neg_axes = tuple([x - ndim for x in axes])
                check(arrs[i], axes)
                check(arrs[i], neg_axes)

        @from_generic([transpose_issue_4708])
        def check_issue_4708(pyfunc, m, n):
            expected = pyfunc.py_func(m, n)
            got = pyfunc(m, n)
            np.testing.assert_equal(got, expected)
        check_issue_4708(3, 2)
        check_issue_4708(2, 3)
        check_issue_4708(5, 4)
        self.disable_leak_check()
        check_err_invalid_args(arrs[1], 'foo')
        check_err_invalid_args(arrs[1], ('foo',))
        check_err_invalid_args(arrs[1], 5.3)
        check_err_invalid_args(arrs[2], (1.2, 5))
        check_err_axis_repeated(arrs[1], (0, 0))
        check_err_axis_repeated(arrs[2], (2, 0, 0))
        check_err_axis_repeated(arrs[3], (3, 2, 1, 1))
        check_err_axis_oob(arrs[0], (1,))
        check_err_axis_oob(arrs[0], (-2,))
        check_err_axis_oob(arrs[1], (0, 2))
        check_err_axis_oob(arrs[1], (-3, 2))
        check_err_axis_oob(arrs[1], (0, -3))
        check_err_axis_oob(arrs[2], (3, 1, 2))
        check_err_axis_oob(arrs[2], (-4, 1, 2))
        check_err_axis_oob(arrs[3], (3, 1, 2, 5))
        check_err_axis_oob(arrs[3], (3, 1, 2, -5))
        with self.assertRaises(TypingError) as e:
            jit(nopython=True)(numpy_transpose_array)((np.array([0, 1]),))
        self.assertIn('np.transpose does not accept tuples', str(e.exception))

    def test_numpy_resize_basic(self):
        pyfunc = numpy_resize
        cfunc = njit(pyfunc)

        def inputs():
            yield (np.array([[1, 2], [3, 4]]), (2, 4))
            yield (np.array([[1, 2], [3, 4]]), (4, 2))
            yield (np.array([[1, 2], [3, 4]]), (4, 3))
            yield (np.array([[1, 2], [3, 4]]), (0,))
            yield (np.array([[1, 2], [3, 4]]), (0, 2))
            yield (np.array([[1, 2], [3, 4]]), (2, 0))
            yield (np.zeros(0, dtype=float), (2, 1))
            yield (np.array([[1, 2], [3, 4]]), (4,))
            yield (np.array([[1, 2], [3, 4]]), 4)
            yield (np.zeros((1, 3), dtype=int), (2, 1))
            yield (np.array([], dtype=float), (4, 2))
            yield ([0, 1, 2, 3], (2, 3))
            yield (4, (2, 3))
        for a, new_shape in inputs():
            self.assertPreciseEqual(pyfunc(a, new_shape), cfunc(a, new_shape))

    def test_numpy_resize_exception(self):
        self.disable_leak_check()
        cfunc = njit(numpy_resize)
        with self.assertRaises(TypingError) as raises:
            cfunc('abc', (2, 3))
        self.assertIn('The argument "a" must be array-like', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc(np.array([[0, 1], [2, 3]]), 'abc')
        self.assertIn('The argument "new_shape" must be an integer or a tuple of integers', str(raises.exception))
        with self.assertRaises(ValueError) as raises:
            cfunc(np.array([[0, 1], [2, 3]]), (-2, 3))
        self.assertIn('All elements of `new_shape` must be non-negative', str(raises.exception))
        with self.assertRaises(ValueError) as raises:
            cfunc(np.array([[0, 1], [2, 3]]), -4)
        self.assertIn('All elements of `new_shape` must be non-negative', str(raises.exception))

    def test_expand_dims(self):
        pyfunc = expand_dims
        cfunc = njit(pyfunc)

        def check(arr, axis):
            expected = pyfunc(arr, axis)
            self.memory_leak_setup()
            got = cfunc(arr, axis)
            self.assertPreciseEqual(got, expected)
            del got
            self.memory_leak_teardown()

        def check_all_axes(arr):
            for axis in range(-arr.ndim - 1, arr.ndim + 1):
                check(arr, axis)
        arr = np.arange(5)
        check_all_axes(arr)
        arr = np.arange(24).reshape((2, 3, 4))
        check_all_axes(arr)
        check_all_axes(arr.T)
        check_all_axes(arr[::-1])
        arr = np.array(42)
        check_all_axes(arr)

    def test_expand_dims_exceptions(self):
        pyfunc = expand_dims
        cfunc = jit(nopython=True)(pyfunc)
        arr = np.arange(5)
        with self.assertTypingError() as raises:
            cfunc('hello', 3)
        self.assertIn('First argument "a" must be an array', str(raises.exception))
        with self.assertTypingError() as raises:
            cfunc(arr, 'hello')
        self.assertIn('Argument "axis" must be an integer', str(raises.exception))

    def check_atleast_nd(self, pyfunc, cfunc):

        def check_result(got, expected):
            self.assertStridesEqual(got, expected)
            self.assertPreciseEqual(got.flatten(), expected.flatten())

        def check_single(arg):
            check_result(cfunc(arg), pyfunc(arg))

        def check_tuple(*args):
            expected_tuple = pyfunc(*args)
            got_tuple = cfunc(*args)
            self.assertEqual(len(got_tuple), len(expected_tuple))
            for got, expected in zip(got_tuple, expected_tuple):
                check_result(got, expected)
        a1 = np.array(42)
        a2 = np.array(5j)
        check_single(a1)
        check_tuple(a1, a2)
        b1 = np.arange(5)
        b2 = np.arange(6) + 1j
        b3 = b1[::-1]
        check_single(b1)
        check_tuple(b1, b2, b3)
        c1 = np.arange(6).reshape((2, 3))
        c2 = c1.T
        c3 = c1[::-1]
        check_single(c1)
        check_tuple(c1, c2, c3)
        d1 = np.arange(24).reshape((2, 3, 4))
        d2 = d1.T
        d3 = d1[::-1]
        check_single(d1)
        check_tuple(d1, d2, d3)
        e = np.arange(16).reshape((2, 2, 2, 2))
        check_single(e)
        check_tuple(a1, b2, c3, d2)

    def test_atleast_1d(self):
        pyfunc = atleast_1d
        cfunc = jit(nopython=True)(pyfunc)
        self.check_atleast_nd(pyfunc, cfunc)

    def test_atleast_2d(self):
        pyfunc = atleast_2d
        cfunc = jit(nopython=True)(pyfunc)
        self.check_atleast_nd(pyfunc, cfunc)

    def test_atleast_3d(self):
        pyfunc = atleast_3d
        cfunc = jit(nopython=True)(pyfunc)
        self.check_atleast_nd(pyfunc, cfunc)

    def check_as_strided(self, pyfunc):
        cfunc = njit(pyfunc)

        def check(arr):
            expected = pyfunc(arr)
            got = cfunc(arr)
            self.assertPreciseEqual(got, expected)
        arr = np.arange(24)
        check(arr)
        check(arr.reshape((6, 4)))
        check(arr.reshape((4, 1, 6)))

    def test_as_strided(self):
        self.check_as_strided(as_strided1)
        self.check_as_strided(as_strided2)

    def test_sliding_window_view(self):

        def check(arr, window_shape, axis):
            expected = np.lib.stride_tricks.sliding_window_view(arr, window_shape, axis, writeable=True)
            got = sliding_window_view(arr, window_shape, axis)
            self.assertPreciseEqual(got, expected)
        arr1 = np.arange(24)
        for axis in [None, 0, -1, (0,)]:
            with self.subTest(f'1d array, axis={axis}'):
                check(arr1, 5, axis)
        arr2 = np.arange(200).reshape(10, 20)
        for axis in [0, -1]:
            with self.subTest(f'2d array, axis={axis}'):
                check(arr2, 5, axis)
        for axis in [None, (0, 1), (1, 0), (1, -2)]:
            with self.subTest(f'2d array, axis={axis}'):
                check(arr2, (5, 8), axis)
        arr4 = np.arange(200).reshape(4, 5, 5, 2)
        for axis in [(1, 2), (-2, -3)]:
            with self.subTest(f'4d array, axis={axis}'):
                check(arr4, (3, 2), axis)
        with self.subTest('2d array, repeated axes'):
            check(arr2, (5, 3, 3), (0, 1, 0))

    def test_sliding_window_view_errors(self):

        def _raises(msg, *args):
            with self.assertRaises(ValueError) as raises:
                sliding_window_view(*args)
            self.assertIn(msg, str(raises.exception))

        def _typing_error(msg, *args):
            with self.assertRaises(errors.TypingError) as raises:
                sliding_window_view(*args)
            self.assertIn(msg, str(raises.exception))
        self.disable_leak_check()
        arr1 = np.arange(24)
        arr2 = np.arange(200).reshape(10, 20)
        with self.subTest('1d window shape too large'):
            _raises('window_shape cannot be larger', arr1, 25, None)
        with self.subTest('2d window shape too large'):
            _raises('window_shape cannot be larger', arr2, (4, 21), None)
        with self.subTest('1d window negative size'):
            _raises('`window_shape` cannot contain negative', arr1, -1, None)
        with self.subTest('2d window with a negative size'):
            _raises('`window_shape` cannot contain negative', arr2, (4, -3), None)
        with self.subTest('1d array, 2d window shape'):
            _raises('matching length window_shape and axis', arr1, (10, 2), None)
        with self.subTest('2d window shape, only one axis given'):
            _raises('matching length window_shape and axis', arr2, (10, 2), 1)
        with self.subTest('1d window shape, 2 axes given'):
            _raises('matching length window_shape and axis', arr1, 5, (0, 0))
        with self.subTest('1d array, second axis'):
            _raises('Argument axis out of bounds', arr1, 4, 1)
        with self.subTest('1d array, axis -2'):
            _raises('Argument axis out of bounds', arr1, 4, -2)
        with self.subTest('2d array, fourth axis'):
            _raises('Argument axis out of bounds', arr2, (4, 4), (0, 3))
        with self.subTest('2d array, axis -3'):
            _raises('Argument axis out of bounds', arr2, (4, 4), (0, -3))
        with self.subTest('window_shape=None'):
            _typing_error('window_shape must be an integer or tuple of integer', arr1, None)
        with self.subTest('window_shape=float'):
            _typing_error('window_shape must be an integer or tuple of integer', arr1, 3.1)
        with self.subTest('window_shape=tuple(float)'):
            _typing_error('window_shape must be an integer or tuple of integer', arr1, (3.1,))
        with self.subTest('axis=float'):
            _typing_error('axis must be None, an integer or tuple of integer', arr1, 4, 3.1)
        with self.subTest('axis=tuple(float)'):
            _typing_error('axis must be None, an integer or tuple of integer', arr1, 4, (3.1,))

    def test_flatten_array(self, flags=enable_pyobj_flags, layout='C'):
        a = np.arange(9).reshape(3, 3)
        if layout == 'F':
            a = a.T
        pyfunc = flatten_array
        arraytype1 = typeof(a)
        if layout == 'A':
            arraytype1 = arraytype1.copy(layout='A')
        self.assertEqual(arraytype1.layout, layout)
        cfunc = jit((arraytype1,), **flags)(pyfunc)
        expected = pyfunc(a)
        got = cfunc(a)
        np.testing.assert_equal(expected, got)

    def test_flatten_array_npm(self):
        self.test_flatten_array(flags=no_pyobj_flags)
        self.test_flatten_array(flags=no_pyobj_flags, layout='F')
        self.test_flatten_array(flags=no_pyobj_flags, layout='A')

    def test_ravel_array(self, flags=enable_pyobj_flags):

        def generic_check(pyfunc, a, assume_layout):
            arraytype1 = typeof(a)
            self.assertEqual(arraytype1.layout, assume_layout)
            cfunc = jit((arraytype1,), **flags)(pyfunc)
            expected = pyfunc(a)
            got = cfunc(a)
            np.testing.assert_equal(expected, got)
            py_copied = a.ctypes.data != expected.ctypes.data
            nb_copied = a.ctypes.data != got.ctypes.data
            self.assertEqual(py_copied, assume_layout != 'C')
            self.assertEqual(py_copied, nb_copied)
        check_method = partial(generic_check, ravel_array)
        check_function = partial(generic_check, numpy_ravel_array)

        def check(*args, **kwargs):
            check_method(*args, **kwargs)
            check_function(*args, **kwargs)
        check(np.arange(9).reshape(3, 3), assume_layout='C')
        check(np.arange(9).reshape(3, 3, order='F'), assume_layout='F')
        check(np.arange(18).reshape(3, 3, 2)[:, :, 0], assume_layout='A')
        check(np.arange(18).reshape(2, 3, 3), assume_layout='C')
        check(np.arange(18).reshape(2, 3, 3, order='F'), assume_layout='F')
        check(np.arange(36).reshape(2, 3, 3, 2)[:, :, :, 0], assume_layout='A')

    def test_ravel_array_size(self, flags=enable_pyobj_flags):
        a = np.arange(9).reshape(3, 3)
        pyfunc = ravel_array_size
        arraytype1 = typeof(a)
        cfunc = jit((arraytype1,), **flags)(pyfunc)
        expected = pyfunc(a)
        got = cfunc(a)
        np.testing.assert_equal(expected, got)

    def test_ravel_array_npm(self):
        self.test_ravel_array(flags=no_pyobj_flags)

    def test_ravel_array_size_npm(self):
        self.test_ravel_array_size(flags=no_pyobj_flags)

    def test_transpose_array(self, flags=enable_pyobj_flags):

        @from_generic([transpose_array, numpy_transpose_array])
        def check(pyfunc):
            a = np.arange(9).reshape(3, 3)
            arraytype1 = typeof(a)
            cfunc = jit((arraytype1,), **flags)(pyfunc)
            expected = pyfunc(a)
            got = cfunc(a)
            np.testing.assert_equal(expected, got)
        check()

    def test_transpose_array_npm(self):
        self.test_transpose_array(flags=no_pyobj_flags)

    def test_squeeze_array(self, flags=enable_pyobj_flags):
        a = np.arange(2 * 1 * 3 * 1 * 4).reshape(2, 1, 3, 1, 4)
        pyfunc = squeeze_array
        arraytype1 = typeof(a)
        cfunc = jit((arraytype1,), **flags)(pyfunc)
        expected = pyfunc(a)
        got = cfunc(a)
        np.testing.assert_equal(expected, got)

    def test_squeeze_array_npm(self):
        with self.assertRaises(errors.TypingError) as raises:
            self.test_squeeze_array(flags=no_pyobj_flags)
        self.assertIn('squeeze', str(raises.exception))

    def test_add_axis(self):

        @njit
        def np_new_axis_getitem(a, idx):
            return a[idx]

        @njit
        def np_new_axis_setitem(a, idx, item):
            a[idx] = item
            return a
        a = np.arange(4 * 5 * 6 * 7).reshape((4, 5, 6, 7))
        idx_cases = [(slice(None), np.newaxis), (np.newaxis, slice(None)), (slice(1), np.newaxis, 1), (np.newaxis, 2, slice(None)), (slice(1), Ellipsis, np.newaxis, 1), (1, np.newaxis, Ellipsis), (np.newaxis, slice(1), np.newaxis, 1), (1, Ellipsis, None, np.newaxis), (np.newaxis, slice(1), Ellipsis, np.newaxis, 1), (1, np.newaxis, np.newaxis, Ellipsis), (np.newaxis, 1, np.newaxis, Ellipsis), (slice(3), 1, np.newaxis, None), (np.newaxis, 1, Ellipsis, None)]
        pyfunc_getitem = np_new_axis_getitem.py_func
        cfunc_getitem = np_new_axis_getitem
        pyfunc_setitem = np_new_axis_setitem.py_func
        cfunc_setitem = np_new_axis_setitem
        for idx in idx_cases:
            expected = pyfunc_getitem(a, idx)
            got = cfunc_getitem(a, idx)
            np.testing.assert_equal(expected, got)
            a_empty = np.zeros_like(a)
            item = a[idx]
            expected = pyfunc_setitem(a_empty.copy(), idx, item)
            got = cfunc_setitem(a_empty.copy(), idx, item)
            np.testing.assert_equal(expected, got)

    def test_bad_index_npm(self):
        with self.assertTypingError() as raises:
            arraytype1 = from_dtype(np.dtype([('x', np.int32), ('y', np.int32)]))
            arraytype2 = types.Array(types.int32, 2, 'C')
            njit((arraytype1, arraytype2))(bad_index)
        self.assertIn('Unsupported array index type', str(raises.exception))

    def test_bad_float_index_npm(self):
        with self.assertTypingError() as raises:
            njit((types.Array(types.float64, 2, 'C'),))(bad_float_index)
        self.assertIn('Unsupported array index type float64', str(raises.exception))

    def test_fill_diagonal_basic(self):
        pyfunc = numpy_fill_diagonal
        cfunc = jit(nopython=True)(pyfunc)

        def _shape_variations(n):
            yield (n, n)
            yield (2 * n, n)
            yield (n, 2 * n)
            yield (2 * n + 1, 2 * n - 1)
            yield (n, n, n, n)
            yield (1, 1, 1)

        def _val_variations():
            yield 1
            yield 3.142
            yield np.nan
            yield (-np.inf)
            yield True
            yield np.arange(4)
            yield (4,)
            yield [8, 9]
            yield np.arange(54).reshape(9, 3, 2, 1)
            yield np.asfortranarray(np.arange(9).reshape(3, 3))
            yield np.arange(9).reshape(3, 3)[::-1]

        def _multi_dimensional_array_variations(n):
            for shape in _shape_variations(n):
                yield np.zeros(shape, dtype=np.float64)
                yield np.asfortranarray(np.ones(shape, dtype=np.float64))

        def _multi_dimensional_array_variations_strided(n):
            for shape in _shape_variations(n):
                tmp = np.zeros(tuple([x * 2 for x in shape]), dtype=np.float64)
                slicer = tuple((slice(0, x * 2, 2) for x in shape))
                yield tmp[slicer]

        def _check_fill_diagonal(arr, val):
            for wrap in (None, True, False):
                a = arr.copy()
                b = arr.copy()
                if wrap is None:
                    params = {}
                else:
                    params = {'wrap': wrap}
                pyfunc(a, val, **params)
                cfunc(b, val, **params)
                self.assertPreciseEqual(a, b)
        for arr in _multi_dimensional_array_variations(3):
            for val in _val_variations():
                _check_fill_diagonal(arr, val)
        for arr in _multi_dimensional_array_variations_strided(3):
            for val in _val_variations():
                _check_fill_diagonal(arr, val)
        arr = np.array([True] * 9).reshape(3, 3)
        _check_fill_diagonal(arr, False)
        _check_fill_diagonal(arr, [False, True, False])
        _check_fill_diagonal(arr, np.array([True, False, True]))

    def test_fill_diagonal_exception_cases(self):
        pyfunc = numpy_fill_diagonal
        cfunc = jit(nopython=True)(pyfunc)
        val = 1
        self.disable_leak_check()
        for a in (np.array([]), np.ones(5)):
            with self.assertRaises(TypingError) as raises:
                cfunc(a, val)
            assert 'The first argument must be at least 2-D' in str(raises.exception)
        with self.assertRaises(ValueError) as raises:
            a = np.zeros((3, 3, 4))
            cfunc(a, val)
            self.assertEqual('All dimensions of input must be of equal length', str(raises.exception))

        def _assert_raises(arr, val):
            with self.assertRaises(ValueError) as raises:
                cfunc(arr, val)
            self.assertEqual('Unable to safely conform val to a.dtype', str(raises.exception))
        arr = np.zeros((3, 3), dtype=np.int32)
        val = np.nan
        _assert_raises(arr, val)
        val = [3.3, np.inf]
        _assert_raises(arr, val)
        val = np.array([1, 2, 10000000000.0], dtype=np.int64)
        _assert_raises(arr, val)
        arr = np.zeros((3, 3), dtype=np.float32)
        val = [1.4, 2.6, -1e+100]
        _assert_raises(arr, val)
        val = 1.1e+100
        _assert_raises(arr, val)
        val = np.array([-1e+100])
        _assert_raises(arr, val)

    def test_broadcast_to(self):
        pyfunc = numpy_broadcast_to
        cfunc = jit(nopython=True)(pyfunc)
        data = [[np.array(0), (0,)], [np.array(0), (1,)], [np.array(0), (3,)], [np.ones(1), (1,)], [np.ones(1), (2,)], [np.ones(1), (1, 2, 3)], [np.arange(3), (3,)], [np.arange(3), (1, 3)], [np.arange(3), (2, 3)], [np.ones(0), 0], [np.ones(1), 1], [np.ones(1), 2], [np.ones(1), (0,)], [np.ones((1, 2)), (0, 2)], [np.ones((2, 1)), (2, 0)], [2, (2, 2)], [(1, 2), (2, 2)]]
        for input_array, shape in data:
            expected = pyfunc(input_array, shape)
            got = cfunc(input_array, shape)
            self.assertPreciseEqual(got, expected)

    def test_broadcast_to_0d_array(self):
        pyfunc = numpy_broadcast_to
        cfunc = jit(nopython=True)(pyfunc)
        inputs = [np.array(123), 123, True]
        shape = ()
        for arr in inputs:
            expected = pyfunc(arr, shape)
            got = cfunc(arr, shape)
            self.assertPreciseEqual(expected, got)
            self.assertFalse(got.flags['WRITEABLE'])

    def test_broadcast_to_raises(self):
        pyfunc = numpy_broadcast_to
        cfunc = jit(nopython=True)(pyfunc)
        data = [[np.zeros((0,)), (), TypingError, 'Cannot broadcast a non-scalar to a scalar array'], [np.zeros((1,)), (), TypingError, 'Cannot broadcast a non-scalar to a scalar array'], [np.zeros((3,)), (), TypingError, 'Cannot broadcast a non-scalar to a scalar array'], [(), (), TypingError, 'Cannot broadcast a non-scalar to a scalar array'], [(123,), (), TypingError, 'Cannot broadcast a non-scalar to a scalar array'], [np.zeros((3,)), (1,), ValueError, 'operands could not be broadcast together with remapped shapes'], [np.zeros((3,)), (2,), ValueError, 'operands could not be broadcast together with remapped shapes'], [np.zeros((3,)), (4,), ValueError, 'operands could not be broadcast together with remapped shapes'], [np.zeros((1, 2)), (2, 1), ValueError, 'operands could not be broadcast together with remapped shapes'], [np.zeros((1, 1)), (1,), ValueError, 'input operand has more dimensions than allowed by the axis remapping'], [np.zeros((2, 2)), (3,), ValueError, 'input operand has more dimensions than allowed by the axis remapping'], [np.zeros((1,)), -1, ValueError, 'all elements of broadcast shape must be non-negative'], [np.zeros((1,)), (-1,), ValueError, 'all elements of broadcast shape must be non-negative'], [np.zeros((1, 2)), (-1, 2), ValueError, 'all elements of broadcast shape must be non-negative'], [np.zeros((1, 2)), (1.1, 2.2), TypingError, 'The second argument "shape" must be a tuple of integers'], ['hello', (3,), TypingError, 'The first argument "array" must be array-like'], [3, (2, 'a'), TypingError, 'object cannot be interpreted as an integer']]
        self.disable_leak_check()
        for arr, target_shape, err, msg in data:
            with self.assertRaises(err) as raises:
                cfunc(arr, target_shape)
            self.assertIn(msg, str(raises.exception))

    def test_broadcast_to_corner_cases(self):

        @njit
        def _broadcast_to_1():
            return np.broadcast_to('a', (2, 3))
        expected = _broadcast_to_1.py_func()
        got = _broadcast_to_1()
        self.assertPreciseEqual(expected, got)

    def test_broadcast_to_change_view(self):
        pyfunc = numpy_broadcast_to
        cfunc = jit(nopython=True)(pyfunc)
        input_array = np.zeros(2, dtype=np.int32)
        shape = (2, 2)
        view = cfunc(input_array, shape)
        input_array[0] = 10
        self.assertEqual(input_array.sum(), 10)
        self.assertEqual(view.sum(), 20)

    def test_broadcast_to_indexing(self):
        pyfunc = numpy_broadcast_to_indexing
        cfunc = jit(nopython=True)(pyfunc)
        data = [[np.ones(2), (2, 2), (1,)]]
        for input_array, shape, idx in data:
            expected = pyfunc(input_array, shape, idx)
            got = cfunc(input_array, shape, idx)
            self.assertPreciseEqual(got, expected)

    def test_broadcast_to_array_attrs(self):

        @njit
        def foo(arr):
            ret = np.broadcast_to(arr, (2, 3))
            return (ret, ret.size, ret.shape, ret.strides)
        arr = np.arange(3)
        expected = foo.py_func(arr)
        got = foo(arr)
        self.assertPreciseEqual(expected, got)

    def test_broadcast_shapes(self):
        pyfunc = numpy_broadcast_shapes
        cfunc = jit(nopython=True)(pyfunc)
        data = [[()], [(), ()], [(7,)], [(1, 2)], [(1, 1)], [(1, 1), (3, 4)], [(6, 7), (5, 6, 1), (7,), (5, 1, 7)], [(5, 6, 1)], [(1, 3), (3, 1)], [(1, 0), (0, 0)], [(0, 1), (0, 0)], [(1, 0), (0, 1)], [(1, 1), (0, 0)], [(1, 1), (1, 0)], [(1, 1), (0, 1)], [(), (0,)], [(0,), (0, 0)], [(0,), (0, 1)], [(1,), (0, 0)], [(), (0, 0)], [(1, 1), (0,)], [(1,), (0, 1)], [(1,), (1, 0)], [(), (1, 0)], [(), (0, 1)], [(1,), (3,)], [2, (3, 2)]]
        for input_shape in data:
            expected = pyfunc(*input_shape)
            got = cfunc(*input_shape)
            self.assertIsInstance(got, tuple)
            self.assertPreciseEqual(expected, got)

    def test_broadcast_shapes_raises(self):
        pyfunc = numpy_broadcast_shapes
        cfunc = jit(nopython=True)(pyfunc)
        self.disable_leak_check()
        data = [[(3,), (4,)], [(2, 3), (2,)], [(3,), (3,), (4,)], [(1, 3, 4), (2, 3, 3)], [(1, 2), (3, 1), (3, 2), (10, 5)], [2, (2, 3)]]
        for input_shape in data:
            with self.assertRaises(ValueError) as raises:
                cfunc(*input_shape)
            self.assertIn('shape mismatch: objects cannot be broadcast to a single shape', str(raises.exception))

    def test_broadcast_shapes_negative_dimension(self):
        pyfunc = numpy_broadcast_shapes
        cfunc = jit(nopython=True)(pyfunc)
        self.disable_leak_check()
        with self.assertRaises(ValueError) as raises:
            cfunc((1, 2), 2, -2)
        self.assertIn('negative dimensions are not allowed', str(raises.exception))

    def test_broadcast_shapes_invalid_type(self):
        pyfunc = numpy_broadcast_shapes
        cfunc = jit(nopython=True)(pyfunc)
        self.disable_leak_check()
        inps = [((1, 2), ('hello',)), (3.4,), ('string',), (1.2, 'a'), (1, (1.2, 'a'))]
        for inp in inps:
            with self.assertRaises(TypingError) as raises:
                cfunc(*inp)
            self.assertIn('must be either an int or tuple[int]', str(raises.exception))

    def test_shape(self):
        pyfunc = numpy_shape
        cfunc = jit(nopython=True)(pyfunc)

        def check(x):
            expected = pyfunc(x)
            got = cfunc(x)
            self.assertPreciseEqual(got, expected)
        for t in [(), (1,), (2, 3), (4, 5, 6)]:
            arr = np.empty(t)
            check(arr)
        for t in [1, False, [1], [[1, 2], [3, 4]], (1,), (1, 2, 3)]:
            check(arr)
        with self.assertRaises(TypingError) as raises:
            cfunc('a')
        self.assertIn('The argument to np.shape must be array-like', str(raises.exception))

    def test_flatnonzero_basic(self):
        pyfunc = numpy_flatnonzero
        cfunc = jit(nopython=True)(pyfunc)

        def a_variations():
            yield np.arange(-5, 5)
            yield np.full(5, fill_value=0)
            yield np.array([])
            a = self.random.randn(100)
            a[np.abs(a) > 0.2] = 0.0
            yield a
            yield a.reshape(5, 5, 4)
            yield a.reshape(50, 2, order='F')
            yield a.reshape(25, 4)[1::2]
            yield (a * 1j)
        for a in a_variations():
            expected = pyfunc(a)
            got = cfunc(a)
            self.assertPreciseEqual(expected, got)

    def test_argwhere_basic(self):
        pyfunc = numpy_argwhere
        cfunc = jit(nopython=True)(pyfunc)

        def a_variations():
            yield (np.arange(-5, 5) > 2)
            yield np.full(5, fill_value=0)
            yield np.full(5, fill_value=1)
            yield np.array([])
            yield np.array([-1.0, 0.0, 1.0])
            a = self.random.randn(100)
            yield (a > 0.2)
            yield (a.reshape(5, 5, 4) > 0.5)
            yield (a.reshape(50, 2, order='F') > 0.5)
            yield (a.reshape(25, 4)[1::2] > 0.5)
            yield (a == a - 1)
            yield (a > -a)
        for a in a_variations():
            expected = pyfunc(a)
            got = cfunc(a)
            self.assertPreciseEqual(expected, got)

    @staticmethod
    def array_like_variations():
        yield ((1.1, 2.2), (3.3, 4.4), (5.5, 6.6))
        yield (0.0, 1.0, 0.0, -6.0)
        yield ([0, 1], [2, 3])
        yield ()
        yield np.nan
        yield 0
        yield 1
        yield False
        yield True
        yield (True, False, True)
        yield (2 + 1j)
        yield None
        yield 'a_string'
        yield ''

    def test_flatnonzero_array_like(self):
        pyfunc = numpy_flatnonzero
        cfunc = jit(nopython=True)(pyfunc)
        for a in self.array_like_variations():
            expected = pyfunc(a)
            got = cfunc(a)
            self.assertPreciseEqual(expected, got)

    def test_argwhere_array_like(self):
        pyfunc = numpy_argwhere
        cfunc = jit(nopython=True)(pyfunc)
        for a in self.array_like_variations():
            expected = pyfunc(a)
            got = cfunc(a)
            self.assertPreciseEqual(expected, got)

    def broadcast_arrays_assert_correct_shape(self, input_shapes, expected_shape):
        pyfunc = numpy_broadcast_arrays
        cfunc = jit(nopython=True)(pyfunc)
        inarrays = [np.zeros(s) for s in input_shapes]
        outarrays = cfunc(*inarrays)
        expected = [expected_shape] * len(inarrays)
        got = [a.shape for a in outarrays]
        self.assertPreciseEqual(expected, got)

    def test_broadcast_arrays_same_input_shapes(self):
        pyfunc = numpy_broadcast_arrays
        cfunc = jit(nopython=True)(pyfunc)
        data = [(1,), (3,), (0, 1), (0, 3), (1, 0), (3, 0), (1, 3), (3, 1), (3, 3)]
        for shape in data:
            input_shapes = [shape]
            self.broadcast_arrays_assert_correct_shape(input_shapes, shape)
            input_shapes2 = [shape, shape]
            self.broadcast_arrays_assert_correct_shape(input_shapes2, shape)
            input_shapes3 = [shape, shape, shape]
            self.broadcast_arrays_assert_correct_shape(input_shapes3, shape)

    def test_broadcast_arrays_two_compatible_by_ones_input_shapes(self):
        data = [[[(1,), (3,)], (3,)], [[(1, 3), (3, 3)], (3, 3)], [[(3, 1), (3, 3)], (3, 3)], [[(1, 3), (3, 1)], (3, 3)], [[(1, 1), (3, 3)], (3, 3)], [[(1, 1), (1, 3)], (1, 3)], [[(1, 1), (3, 1)], (3, 1)], [[(1, 0), (0, 0)], (0, 0)], [[(0, 1), (0, 0)], (0, 0)], [[(1, 0), (0, 1)], (0, 0)], [[(1, 1), (0, 0)], (0, 0)], [[(1, 1), (1, 0)], (1, 0)], [[(1, 1), (0, 1)], (0, 1)]]
        for input_shapes, expected_shape in data:
            self.broadcast_arrays_assert_correct_shape(input_shapes, expected_shape)
            self.broadcast_arrays_assert_correct_shape(input_shapes[::-1], expected_shape)

    def test_broadcast_arrays_two_compatible_by_prepending_ones_input_shapes(self):
        data = [[[(), (3,)], (3,)], [[(3,), (3, 3)], (3, 3)], [[(3,), (3, 1)], (3, 3)], [[(1,), (3, 3)], (3, 3)], [[(), (3, 3)], (3, 3)], [[(1, 1), (3,)], (1, 3)], [[(1,), (3, 1)], (3, 1)], [[(1,), (1, 3)], (1, 3)], [[(), (1, 3)], (1, 3)], [[(), (3, 1)], (3, 1)], [[(), (0,)], (0,)], [[(0,), (0, 0)], (0, 0)], [[(0,), (0, 1)], (0, 0)], [[(1,), (0, 0)], (0, 0)], [[(), (0, 0)], (0, 0)], [[(1, 1), (0,)], (1, 0)], [[(1,), (0, 1)], (0, 1)], [[(1,), (1, 0)], (1, 0)], [[(), (1, 0)], (1, 0)], [[(), (0, 1)], (0, 1)]]
        for input_shapes, expected_shape in data:
            self.broadcast_arrays_assert_correct_shape(input_shapes, expected_shape)
            self.broadcast_arrays_assert_correct_shape(input_shapes[::-1], expected_shape)

    def test_broadcast_arrays_scalar_input(self):
        pyfunc = numpy_broadcast_arrays
        cfunc = jit(nopython=True)(pyfunc)
        data = [[[True, False], (1,)], [[1, 2], (1,)], [[(1, 2), 2], (2,)]]
        for inarrays, expected_shape in data:
            outarrays = cfunc(*inarrays)
            got = [a.shape for a in outarrays]
            expected = [expected_shape] * len(inarrays)
            self.assertPreciseEqual(expected, got)

    def test_broadcast_arrays_tuple_input(self):
        pyfunc = numpy_broadcast_arrays
        cfunc = jit(nopython=True)(pyfunc)
        outarrays = cfunc((123, 456), (789,))
        expected = [(2,), (2,)]
        got = [a.shape for a in outarrays]
        self.assertPreciseEqual(expected, got)

    def test_broadcast_arrays_non_array_input(self):
        pyfunc = numpy_broadcast_arrays
        cfunc = jit(nopython=True)(pyfunc)
        outarrays = cfunc(np.intp(2), np.zeros((1, 3), dtype=np.intp))
        expected = [(1, 3), (1, 3)]
        got = [a.shape for a in outarrays]
        self.assertPreciseEqual(expected, got)

    def test_broadcast_arrays_invalid_mixed_input_types(self):
        pyfunc = numpy_broadcast_arrays
        cfunc = jit(nopython=True)(pyfunc)
        self.disable_leak_check()
        with self.assertRaises(TypingError) as raises:
            arr = np.arange(6).reshape((2, 3))
            b = True
            cfunc(arr, b)
        self.assertIn('Mismatch of argument types', str(raises.exception))

    def test_broadcast_arrays_invalid_input(self):
        pyfunc = numpy_broadcast_arrays
        cfunc = jit(nopython=True)(pyfunc)
        self.disable_leak_check()
        with self.assertRaises(TypingError) as raises:
            arr = np.zeros(3, dtype=np.int64)
            s = 'hello world'
            cfunc(arr, s)
        self.assertIn('Argument "1" must be array-like', str(raises.exception))

    def test_broadcast_arrays_incompatible_shapes_raise_valueerror(self):
        pyfunc = numpy_broadcast_arrays
        cfunc = jit(nopython=True)(pyfunc)
        self.disable_leak_check()
        data = [[(3,), (4,)], [(2, 3), (2,)], [(3,), (3,), (4,)], [(1, 3, 4), (2, 3, 3)]]
        for input_shapes in data:
            for shape in [input_shapes, input_shapes[::-1]]:
                with self.assertRaises(ValueError) as raises:
                    inarrays = [np.zeros(s) for s in shape]
                    cfunc(*inarrays)
                self.assertIn('shape mismatch: objects cannot be broadcast to a single shape', str(raises.exception))

    def test_readonly_after_flatten(self):

        def unfold_flatten(x, y):
            r, c = x.shape
            a = np.broadcast_to(x, (y, r, c))
            b = np.swapaxes(a, 0, 1)
            cc = b.flatten()
            d = np.reshape(cc, (-1, c))
            d[y - 1:, :] = d[:1 - y]
            return d
        pyfunc = unfold_flatten
        cfunc = jit(nopython=True)(pyfunc)
        res_nb = cfunc(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), 2)
        res_py = pyfunc(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), 2)
        np.testing.assert_array_equal(res_py, res_nb)

    def test_readonly_after_ravel(self):

        def unfold_ravel(x, y):
            r, c = x.shape
            a = np.broadcast_to(x, (y, r, c))
            b = np.swapaxes(a, 0, 1)
            cc = b.ravel()
            d = np.reshape(cc, (-1, c))
            d[y - 1:, :] = d[:1 - y]
            return d
        pyfunc = unfold_ravel
        cfunc = jit(nopython=True)(pyfunc)
        res_nb = cfunc(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), 2)
        res_py = pyfunc(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), 2)
        np.testing.assert_array_equal(res_py, res_nb)

    def test_mutability_after_ravel(self):
        self.disable_leak_check()
        a_c = np.arange(9).reshape((3, 3)).copy()
        a_f = a_c.copy(order='F')
        a_c.flags.writeable = False
        a_f.flags.writeable = False

        def try_ravel_w_copy(a):
            result = a.ravel()
            return result
        pyfunc = try_ravel_w_copy
        cfunc = jit(nopython=True)(pyfunc)
        ret_c = cfunc(a_c)
        ret_f = cfunc(a_f)
        msg = 'No copy was performed, so the resulting array must not be writeable'
        self.assertTrue(not ret_c.flags.writeable, msg)
        msg = 'A copy was performed, yet the resulting array is not modifiable'
        self.assertTrue(ret_f.flags.writeable, msg)