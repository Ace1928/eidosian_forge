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
class TestMedian:

    def test_basic(self):
        a0 = np.array(1)
        a1 = np.arange(2)
        a2 = np.arange(6).reshape(2, 3)
        assert_equal(np.median(a0), 1)
        assert_allclose(np.median(a1), 0.5)
        assert_allclose(np.median(a2), 2.5)
        assert_allclose(np.median(a2, axis=0), [1.5, 2.5, 3.5])
        assert_equal(np.median(a2, axis=1), [1, 4])
        assert_allclose(np.median(a2, axis=None), 2.5)
        a = np.array([0.0444502, 0.0463301, 0.141249, 0.0606775])
        assert_almost_equal((a[1] + a[3]) / 2.0, np.median(a))
        a = np.array([0.0463301, 0.0444502, 0.141249])
        assert_equal(a[0], np.median(a))
        a = np.array([0.0444502, 0.141249, 0.0463301])
        assert_equal(a[-1], np.median(a))
        assert_equal(np.median(a).ndim, 0)
        a[1] = np.nan
        assert_equal(np.median(a).ndim, 0)

    def test_axis_keyword(self):
        a3 = np.array([[2, 3], [0, 1], [6, 7], [4, 5]])
        for a in [a3, np.random.randint(0, 100, size=(2, 3, 4))]:
            orig = a.copy()
            np.median(a, axis=None)
            for ax in range(a.ndim):
                np.median(a, axis=ax)
            assert_array_equal(a, orig)
        assert_allclose(np.median(a3, axis=0), [3, 4])
        assert_allclose(np.median(a3.T, axis=1), [3, 4])
        assert_allclose(np.median(a3), 3.5)
        assert_allclose(np.median(a3, axis=None), 3.5)
        assert_allclose(np.median(a3.T), 3.5)

    def test_overwrite_keyword(self):
        a3 = np.array([[2, 3], [0, 1], [6, 7], [4, 5]])
        a0 = np.array(1)
        a1 = np.arange(2)
        a2 = np.arange(6).reshape(2, 3)
        assert_allclose(np.median(a0.copy(), overwrite_input=True), 1)
        assert_allclose(np.median(a1.copy(), overwrite_input=True), 0.5)
        assert_allclose(np.median(a2.copy(), overwrite_input=True), 2.5)
        assert_allclose(np.median(a2.copy(), overwrite_input=True, axis=0), [1.5, 2.5, 3.5])
        assert_allclose(np.median(a2.copy(), overwrite_input=True, axis=1), [1, 4])
        assert_allclose(np.median(a2.copy(), overwrite_input=True, axis=None), 2.5)
        assert_allclose(np.median(a3.copy(), overwrite_input=True, axis=0), [3, 4])
        assert_allclose(np.median(a3.T.copy(), overwrite_input=True, axis=1), [3, 4])
        a4 = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
        np.random.shuffle(a4.ravel())
        assert_allclose(np.median(a4, axis=None), np.median(a4.copy(), axis=None, overwrite_input=True))
        assert_allclose(np.median(a4, axis=0), np.median(a4.copy(), axis=0, overwrite_input=True))
        assert_allclose(np.median(a4, axis=1), np.median(a4.copy(), axis=1, overwrite_input=True))
        assert_allclose(np.median(a4, axis=2), np.median(a4.copy(), axis=2, overwrite_input=True))

    def test_array_like(self):
        x = [1, 2, 3]
        assert_almost_equal(np.median(x), 2)
        x2 = [x]
        assert_almost_equal(np.median(x2), 2)
        assert_allclose(np.median(x2, axis=0), x)

    def test_subclass(self):

        class MySubClass(np.ndarray):

            def __new__(cls, input_array, info=None):
                obj = np.asarray(input_array).view(cls)
                obj.info = info
                return obj

            def mean(self, axis=None, dtype=None, out=None):
                return -7
        a = MySubClass([1, 2, 3])
        assert_equal(np.median(a), -7)

    @pytest.mark.parametrize('arr', ([1.0, 2.0, 3.0], [1.0, np.nan, 3.0], np.nan, 0.0))
    def test_subclass2(self, arr):
        """Check that we return subclasses, even if a NaN scalar."""

        class MySubclass(np.ndarray):
            pass
        m = np.median(np.array(arr).view(MySubclass))
        assert isinstance(m, MySubclass)

    def test_out(self):
        o = np.zeros((4,))
        d = np.ones((3, 4))
        assert_equal(np.median(d, 0, out=o), o)
        o = np.zeros((3,))
        assert_equal(np.median(d, 1, out=o), o)
        o = np.zeros(())
        assert_equal(np.median(d, out=o), o)

    def test_out_nan(self):
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings('always', '', RuntimeWarning)
            o = np.zeros((4,))
            d = np.ones((3, 4))
            d[2, 1] = np.nan
            assert_equal(np.median(d, 0, out=o), o)
            o = np.zeros((3,))
            assert_equal(np.median(d, 1, out=o), o)
            o = np.zeros(())
            assert_equal(np.median(d, out=o), o)

    def test_nan_behavior(self):
        a = np.arange(24, dtype=float)
        a[2] = np.nan
        assert_equal(np.median(a), np.nan)
        assert_equal(np.median(a, axis=0), np.nan)
        a = np.arange(24, dtype=float).reshape(2, 3, 4)
        a[1, 2, 3] = np.nan
        a[1, 1, 2] = np.nan
        assert_equal(np.median(a), np.nan)
        assert_equal(np.median(a).ndim, 0)
        b = np.median(np.arange(24, dtype=float).reshape(2, 3, 4), 0)
        b[2, 3] = np.nan
        b[1, 2] = np.nan
        assert_equal(np.median(a, 0), b)
        b = np.median(np.arange(24, dtype=float).reshape(2, 3, 4), 1)
        b[1, 3] = np.nan
        b[1, 2] = np.nan
        assert_equal(np.median(a, 1), b)
        b = np.median(np.arange(24, dtype=float).reshape(2, 3, 4), (0, 2))
        b[1] = np.nan
        b[2] = np.nan
        assert_equal(np.median(a, (0, 2)), b)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work correctly")
    def test_empty(self):
        a = np.array([], dtype=float)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_equal(np.median(a), np.nan)
            assert_(w[0].category is RuntimeWarning)
            assert_equal(len(w), 2)
        a = np.array([], dtype=float, ndmin=3)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_equal(np.median(a), np.nan)
            assert_(w[0].category is RuntimeWarning)
        b = np.array([], dtype=float, ndmin=2)
        assert_equal(np.median(a, axis=0), b)
        assert_equal(np.median(a, axis=1), b)
        b = np.array(np.nan, dtype=float, ndmin=2)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_equal(np.median(a, axis=2), b)
            assert_(w[0].category is RuntimeWarning)

    def test_object(self):
        o = np.arange(7.0)
        assert_(type(np.median(o.astype(object))), float)
        o[2] = np.nan
        assert_(type(np.median(o.astype(object))), float)

    def test_extended_axis(self):
        o = np.random.normal(size=(71, 23))
        x = np.dstack([o] * 10)
        assert_equal(np.median(x, axis=(0, 1)), np.median(o))
        x = np.moveaxis(x, -1, 0)
        assert_equal(np.median(x, axis=(-2, -1)), np.median(o))
        x = x.swapaxes(0, 1).copy()
        assert_equal(np.median(x, axis=(0, -1)), np.median(o))
        assert_equal(np.median(x, axis=(0, 1, 2)), np.median(x, axis=None))
        assert_equal(np.median(x, axis=(0,)), np.median(x, axis=0))
        assert_equal(np.median(x, axis=(-1,)), np.median(x, axis=-1))
        d = np.arange(3 * 5 * 7 * 11).reshape((3, 5, 7, 11))
        np.random.shuffle(d.ravel())
        assert_equal(np.median(d, axis=(0, 1, 2))[0], np.median(d[:, :, :, 0].flatten()))
        assert_equal(np.median(d, axis=(0, 1, 3))[1], np.median(d[:, :, 1, :].flatten()))
        assert_equal(np.median(d, axis=(3, 1, -4))[2], np.median(d[:, :, 2, :].flatten()))
        assert_equal(np.median(d, axis=(3, 1, 2))[2], np.median(d[2, :, :, :].flatten()))
        assert_equal(np.median(d, axis=(3, 2))[2, 1], np.median(d[2, 1, :, :].flatten()))
        assert_equal(np.median(d, axis=(1, -2))[2, 1], np.median(d[2, :, :, 1].flatten()))
        assert_equal(np.median(d, axis=(1, 3))[2, 2], np.median(d[2, :, 2, :].flatten()))

    def test_extended_axis_invalid(self):
        d = np.ones((3, 5, 7, 11))
        assert_raises(np.AxisError, np.median, d, axis=-5)
        assert_raises(np.AxisError, np.median, d, axis=(0, -5))
        assert_raises(np.AxisError, np.median, d, axis=4)
        assert_raises(np.AxisError, np.median, d, axis=(0, 4))
        assert_raises(ValueError, np.median, d, axis=(1, 1))

    def test_keepdims(self):
        d = np.ones((3, 5, 7, 11))
        assert_equal(np.median(d, axis=None, keepdims=True).shape, (1, 1, 1, 1))
        assert_equal(np.median(d, axis=(0, 1), keepdims=True).shape, (1, 1, 7, 11))
        assert_equal(np.median(d, axis=(0, 3), keepdims=True).shape, (1, 5, 7, 1))
        assert_equal(np.median(d, axis=(1,), keepdims=True).shape, (3, 1, 7, 11))
        assert_equal(np.median(d, axis=(0, 1, 2, 3), keepdims=True).shape, (1, 1, 1, 1))
        assert_equal(np.median(d, axis=(0, 1, 3), keepdims=True).shape, (1, 1, 7, 1))

    @pytest.mark.parametrize(argnames='axis', argvalues=[None, 1, (1,), (0, 1), (-3, -1)])
    def test_keepdims_out(self, axis):
        d = np.ones((3, 5, 7, 11))
        if axis is None:
            shape_out = (1,) * d.ndim
        else:
            axis_norm = normalize_axis_tuple(axis, d.ndim)
            shape_out = tuple((1 if i in axis_norm else d.shape[i] for i in range(d.ndim)))
        out = np.empty(shape_out)
        result = np.median(d, axis=axis, keepdims=True, out=out)
        assert result is out
        assert_equal(result.shape, shape_out)

    @pytest.mark.parametrize('dtype', ['m8[s]'])
    @pytest.mark.parametrize('pos', [0, 23, 10])
    def test_nat_behavior(self, dtype, pos):
        a = np.arange(0, 24, dtype=dtype)
        a[pos] = 'NaT'
        res = np.median(a)
        assert res.dtype == dtype
        assert np.isnat(res)
        res = np.percentile(a, [30, 60])
        assert res.dtype == dtype
        assert np.isnat(res).all()
        a = np.arange(0, 24 * 3, dtype=dtype).reshape(-1, 3)
        a[pos, 1] = 'NaT'
        res = np.median(a, axis=0)
        assert_array_equal(np.isnat(res), [False, True, False])