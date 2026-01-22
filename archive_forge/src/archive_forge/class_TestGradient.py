import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
class TestGradient:

    def test_basic(self):
        v = [[1, 1], [3, 4]]
        x = np.array(v)
        dx = [np.array([[2.0, 3.0], [2.0, 3.0]]), np.array([[0.0, 0.0], [1.0, 1.0]])]
        assert_array_equal(gradient(x), dx)
        assert_array_equal(gradient(v), dx)

    def test_args(self):
        dx = np.cumsum(np.ones(5))
        dx_uneven = [1.0, 2.0, 5.0, 9.0, 11.0]
        f_2d = np.arange(25).reshape(5, 5)
        gradient(np.arange(5), 3.0)
        gradient(np.arange(5), np.array(3.0))
        gradient(np.arange(5), dx)
        gradient(f_2d, 1.5)
        gradient(f_2d, np.array(1.5))
        gradient(f_2d, dx_uneven, dx_uneven)
        gradient(f_2d, dx, 2)
        gradient(f_2d, dx, axis=1)
        assert_raises_regex(ValueError, '.*scalars or 1d', gradient, f_2d, np.stack([dx] * 2, axis=-1), 1)

    def test_badargs(self):
        f_2d = np.arange(25).reshape(5, 5)
        x = np.cumsum(np.ones(5))
        assert_raises(ValueError, gradient, f_2d, x, np.ones(2))
        assert_raises(ValueError, gradient, f_2d, 1, np.ones(2))
        assert_raises(ValueError, gradient, f_2d, np.ones(2), np.ones(2))
        assert_raises(TypeError, gradient, f_2d, x)
        assert_raises(TypeError, gradient, f_2d, x, axis=(0, 1))
        assert_raises(TypeError, gradient, f_2d, x, x, x)
        assert_raises(TypeError, gradient, f_2d, 1, 1, 1)
        assert_raises(TypeError, gradient, f_2d, x, x, axis=1)
        assert_raises(TypeError, gradient, f_2d, 1, 1, axis=1)

    def test_datetime64(self):
        x = np.array(['1910-08-16', '1910-08-11', '1910-08-10', '1910-08-12', '1910-10-12', '1910-12-12', '1912-12-12'], dtype='datetime64[D]')
        dx = np.array([-5, -3, 0, 31, 61, 396, 731], dtype='timedelta64[D]')
        assert_array_equal(gradient(x), dx)
        assert_(dx.dtype == np.dtype('timedelta64[D]'))

    def test_masked(self):
        x = np.ma.array([[1, 1], [3, 4]], mask=[[False, False], [False, False]])
        out = gradient(x)[0]
        assert_equal(type(out), type(x))
        assert_(x._mask is not out._mask)
        x2 = np.ma.arange(5)
        x2[2] = np.ma.masked
        np.gradient(x2, edge_order=2)
        assert_array_equal(x2.mask, [False, False, True, False, False])

    def test_second_order_accurate(self):
        x = np.linspace(0, 1, 10)
        dx = x[1] - x[0]
        y = 2 * x ** 3 + 4 * x ** 2 + 2 * x
        analytical = 6 * x ** 2 + 8 * x + 2
        num_error = np.abs(np.gradient(y, dx, edge_order=2) / analytical - 1)
        assert_(np.all(num_error < 0.03) == True)
        np.random.seed(0)
        x = np.sort(np.random.random(10))
        y = 2 * x ** 3 + 4 * x ** 2 + 2 * x
        analytical = 6 * x ** 2 + 8 * x + 2
        num_error = np.abs(np.gradient(y, x, edge_order=2) / analytical - 1)
        assert_(np.all(num_error < 0.03) == True)

    def test_spacing(self):
        f = np.array([0, 2.0, 3.0, 4.0, 5.0, 5.0])
        f = np.tile(f, (6, 1)) + f.reshape(-1, 1)
        x_uneven = np.array([0.0, 0.5, 1.0, 3.0, 5.0, 7.0])
        x_even = np.arange(6.0)
        fdx_even_ord1 = np.tile([2.0, 1.5, 1.0, 1.0, 0.5, 0.0], (6, 1))
        fdx_even_ord2 = np.tile([2.5, 1.5, 1.0, 1.0, 0.5, -0.5], (6, 1))
        fdx_uneven_ord1 = np.tile([4.0, 3.0, 1.7, 0.5, 0.25, 0.0], (6, 1))
        fdx_uneven_ord2 = np.tile([5.0, 3.0, 1.7, 0.5, 0.25, -0.25], (6, 1))
        for edge_order, exp_res in [(1, fdx_even_ord1), (2, fdx_even_ord2)]:
            res1 = gradient(f, 1.0, axis=(0, 1), edge_order=edge_order)
            res2 = gradient(f, x_even, x_even, axis=(0, 1), edge_order=edge_order)
            res3 = gradient(f, x_even, x_even, axis=None, edge_order=edge_order)
            assert_array_equal(res1, res2)
            assert_array_equal(res2, res3)
            assert_almost_equal(res1[0], exp_res.T)
            assert_almost_equal(res1[1], exp_res)
            res1 = gradient(f, 1.0, axis=0, edge_order=edge_order)
            res2 = gradient(f, x_even, axis=0, edge_order=edge_order)
            assert_(res1.shape == res2.shape)
            assert_almost_equal(res2, exp_res.T)
            res1 = gradient(f, 1.0, axis=1, edge_order=edge_order)
            res2 = gradient(f, x_even, axis=1, edge_order=edge_order)
            assert_(res1.shape == res2.shape)
            assert_array_equal(res2, exp_res)
        for edge_order, exp_res in [(1, fdx_uneven_ord1), (2, fdx_uneven_ord2)]:
            res1 = gradient(f, x_uneven, x_uneven, axis=(0, 1), edge_order=edge_order)
            res2 = gradient(f, x_uneven, x_uneven, axis=None, edge_order=edge_order)
            assert_array_equal(res1, res2)
            assert_almost_equal(res1[0], exp_res.T)
            assert_almost_equal(res1[1], exp_res)
            res1 = gradient(f, x_uneven, axis=0, edge_order=edge_order)
            assert_almost_equal(res1, exp_res.T)
            res1 = gradient(f, x_uneven, axis=1, edge_order=edge_order)
            assert_almost_equal(res1, exp_res)
        res1 = gradient(f, x_even, x_uneven, axis=(0, 1), edge_order=1)
        res2 = gradient(f, x_uneven, x_even, axis=(1, 0), edge_order=1)
        assert_array_equal(res1[0], res2[1])
        assert_array_equal(res1[1], res2[0])
        assert_almost_equal(res1[0], fdx_even_ord1.T)
        assert_almost_equal(res1[1], fdx_uneven_ord1)
        res1 = gradient(f, x_even, x_uneven, axis=(0, 1), edge_order=2)
        res2 = gradient(f, x_uneven, x_even, axis=(1, 0), edge_order=2)
        assert_array_equal(res1[0], res2[1])
        assert_array_equal(res1[1], res2[0])
        assert_almost_equal(res1[0], fdx_even_ord2.T)
        assert_almost_equal(res1[1], fdx_uneven_ord2)

    def test_specific_axes(self):
        v = [[1, 1], [3, 4]]
        x = np.array(v)
        dx = [np.array([[2.0, 3.0], [2.0, 3.0]]), np.array([[0.0, 0.0], [1.0, 1.0]])]
        assert_array_equal(gradient(x, axis=0), dx[0])
        assert_array_equal(gradient(x, axis=1), dx[1])
        assert_array_equal(gradient(x, axis=-1), dx[1])
        assert_array_equal(gradient(x, axis=(1, 0)), [dx[1], dx[0]])
        assert_almost_equal(gradient(x, axis=None), [dx[0], dx[1]])
        assert_almost_equal(gradient(x, axis=None), gradient(x))
        assert_array_equal(gradient(x, 2, 3, axis=(1, 0)), [dx[1] / 2.0, dx[0] / 3.0])
        assert_raises(TypeError, gradient, x, 1, 2, axis=1)
        assert_raises(np.AxisError, gradient, x, axis=3)
        assert_raises(np.AxisError, gradient, x, axis=-3)

    def test_timedelta64(self):
        x = np.array([-5, -3, 10, 12, 61, 321, 300], dtype='timedelta64[D]')
        dx = np.array([2, 7, 7, 25, 154, 119, -21], dtype='timedelta64[D]')
        assert_array_equal(gradient(x), dx)
        assert_(dx.dtype == np.dtype('timedelta64[D]'))

    def test_inexact_dtypes(self):
        for dt in [np.float16, np.float32, np.float64]:
            x = np.array([1, 2, 3], dtype=dt)
            assert_equal(gradient(x).dtype, np.diff(x).dtype)

    def test_values(self):
        gradient(np.arange(2), edge_order=1)
        gradient(np.arange(3), edge_order=2)
        assert_raises(ValueError, gradient, np.arange(0), edge_order=1)
        assert_raises(ValueError, gradient, np.arange(0), edge_order=2)
        assert_raises(ValueError, gradient, np.arange(1), edge_order=1)
        assert_raises(ValueError, gradient, np.arange(1), edge_order=2)
        assert_raises(ValueError, gradient, np.arange(2), edge_order=2)

    @pytest.mark.parametrize('f_dtype', [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_f_decreasing_unsigned_int(self, f_dtype):
        f = np.array([5, 4, 3, 2, 1], dtype=f_dtype)
        g = gradient(f)
        assert_array_equal(g, [-1] * len(f))

    @pytest.mark.parametrize('f_dtype', [np.int8, np.int16, np.int32, np.int64])
    def test_f_signed_int_big_jump(self, f_dtype):
        maxint = np.iinfo(f_dtype).max
        x = np.array([1, 3])
        f = np.array([-1, maxint], dtype=f_dtype)
        dfdx = gradient(f, x)
        assert_array_equal(dfdx, [(maxint + 1) // 2] * 2)

    @pytest.mark.parametrize('x_dtype', [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_x_decreasing_unsigned(self, x_dtype):
        x = np.array([3, 2, 1], dtype=x_dtype)
        f = np.array([0, 2, 4])
        dfdx = gradient(f, x)
        assert_array_equal(dfdx, [-2] * len(x))

    @pytest.mark.parametrize('x_dtype', [np.int8, np.int16, np.int32, np.int64])
    def test_x_signed_int_big_jump(self, x_dtype):
        minint = np.iinfo(x_dtype).min
        maxint = np.iinfo(x_dtype).max
        x = np.array([-1, maxint], dtype=x_dtype)
        f = np.array([minint // 2, 0])
        dfdx = gradient(f, x)
        assert_array_equal(dfdx, [0.5, 0.5])

    def test_return_type(self):
        res = np.gradient(([1, 2], [2, 3]))
        if np._using_numpy2_behavior():
            assert type(res) is tuple
        else:
            assert type(res) is list