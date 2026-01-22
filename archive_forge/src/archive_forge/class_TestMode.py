import os
import re
import warnings
from collections import namedtuple
from itertools import product
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib
from numpy.testing import (assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
import numpy.ma.testutils as mat
from numpy import array, arange, float32, float64, power
import numpy as np
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats._mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from scipy import optimize
from .common_tests import check_named_results
from scipy.spatial.distance import cdist
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
from scipy._lib._util import AxisError
@pytest.mark.filterwarnings('ignore::FutureWarning')
class TestMode:

    def test_empty(self):
        vals, counts = stats.mode([])
        assert_equal(vals, np.array([]))
        assert_equal(counts, np.array([]))

    def test_scalar(self):
        vals, counts = stats.mode(4.0)
        assert_equal(vals, np.array([4.0]))
        assert_equal(counts, np.array([1]))

    def test_basic(self):
        data1 = [3, 5, 1, 10, 23, 3, 2, 6, 8, 6, 10, 6]
        vals = stats.mode(data1)
        assert_equal(vals[0], 6)
        assert_equal(vals[1], 3)

    def test_axes(self):
        data1 = [10, 10, 30, 40]
        data2 = [10, 10, 10, 10]
        data3 = [20, 10, 20, 20]
        data4 = [30, 30, 30, 30]
        data5 = [40, 30, 30, 30]
        arr = np.array([data1, data2, data3, data4, data5])
        vals = stats.mode(arr, axis=None, keepdims=True)
        assert_equal(vals[0], np.array([[30]]))
        assert_equal(vals[1], np.array([[8]]))
        vals = stats.mode(arr, axis=0, keepdims=True)
        assert_equal(vals[0], np.array([[10, 10, 30, 30]]))
        assert_equal(vals[1], np.array([[2, 3, 3, 2]]))
        vals = stats.mode(arr, axis=1, keepdims=True)
        assert_equal(vals[0], np.array([[10], [10], [20], [30], [30]]))
        assert_equal(vals[1], np.array([[2], [4], [3], [4], [3]]))

    @pytest.mark.parametrize('axis', np.arange(-4, 0))
    def test_negative_axes_gh_15375(self, axis):
        np.random.seed(984213899)
        a = np.random.rand(10, 11, 12, 13)
        res0 = stats.mode(a, axis=a.ndim + axis)
        res1 = stats.mode(a, axis=axis)
        np.testing.assert_array_equal(res0, res1)

    def test_mode_result_attributes(self):
        data1 = [3, 5, 1, 10, 23, 3, 2, 6, 8, 6, 10, 6]
        data2 = []
        actual = stats.mode(data1)
        attributes = ('mode', 'count')
        check_named_results(actual, attributes)
        actual2 = stats.mode(data2)
        check_named_results(actual2, attributes)

    def test_mode_nan(self):
        data1 = [3, np.nan, 5, 1, 10, 23, 3, 2, 6, 8, 6, 10, 6]
        actual = stats.mode(data1)
        assert_equal(actual, (6, 3))
        actual = stats.mode(data1, nan_policy='omit')
        assert_equal(actual, (6, 3))
        assert_raises(ValueError, stats.mode, data1, nan_policy='raise')
        assert_raises(ValueError, stats.mode, data1, nan_policy='foobar')

    @pytest.mark.parametrize('data', [[3, 5, 1, 1, 3], [3, np.nan, 5, 1, 1, 3], [3, 5, 1], [3, np.nan, 5, 1]])
    @pytest.mark.parametrize('keepdims', [False, True])
    def test_smallest_equal(self, data, keepdims):
        result = stats.mode(data, nan_policy='omit', keepdims=keepdims)
        if keepdims:
            assert_equal(result[0][0], 1)
        else:
            assert_equal(result[0], 1)

    @pytest.mark.parametrize('axis', np.arange(-3, 3))
    def test_mode_shape_gh_9955(self, axis, dtype=np.float64):
        rng = np.random.default_rng(984213899)
        a = rng.uniform(size=(3, 4, 5)).astype(dtype)
        res = stats.mode(a, axis=axis, keepdims=False)
        reference_shape = list(a.shape)
        reference_shape.pop(axis)
        np.testing.assert_array_equal(res.mode.shape, reference_shape)
        np.testing.assert_array_equal(res.count.shape, reference_shape)

    def test_nan_policy_propagate_gh_9815(self):
        a = [2, np.nan, 1, np.nan]
        res = stats.mode(a)
        assert np.isnan(res.mode) and res.count == 2

    def test_keepdims(self):
        a = np.zeros((1, 2, 3, 0))
        res = stats.mode(a, axis=1, keepdims=False)
        assert res.mode.shape == res.count.shape == (1, 3, 0)
        res = stats.mode(a, axis=1, keepdims=True)
        assert res.mode.shape == res.count.shape == (1, 1, 3, 0)
        a = [[1, 3, 3, np.nan], [1, 1, np.nan, 1]]
        res = stats.mode(a, axis=1, keepdims=False)
        assert_array_equal(res.mode, [3, 1])
        assert_array_equal(res.count, [2, 3])
        res = stats.mode(a, axis=1, keepdims=True)
        assert_array_equal(res.mode, [[3], [1]])
        assert_array_equal(res.count, [[2], [3]])
        a = np.array(a)
        res = stats.mode(a, axis=None, keepdims=False)
        ref = stats.mode(a.ravel(), keepdims=False)
        assert_array_equal(res, ref)
        assert res.mode.shape == ref.mode.shape == ()
        res = stats.mode(a, axis=None, keepdims=True)
        ref = stats.mode(a.ravel(), keepdims=True)
        assert_equal(res.mode.ravel(), ref.mode.ravel())
        assert res.mode.shape == (1, 1)
        assert_equal(res.count.ravel(), ref.count.ravel())
        assert res.count.shape == (1, 1)
        a = [[1, np.nan, np.nan, np.nan, 1], [np.nan, np.nan, np.nan, np.nan, 2], [1, 2, np.nan, 5, 5]]
        res = stats.mode(a, axis=1, keepdims=False, nan_policy='omit')
        assert_array_equal(res.mode, [1, 2, 5])
        assert_array_equal(res.count, [2, 1, 2])
        res = stats.mode(a, axis=1, keepdims=True, nan_policy='omit')
        assert_array_equal(res.mode, [[1], [2], [5]])
        assert_array_equal(res.count, [[2], [1], [2]])
        a = np.array(a)
        res = stats.mode(a, axis=None, keepdims=False, nan_policy='omit')
        ref = stats.mode(a.ravel(), keepdims=False, nan_policy='omit')
        assert_array_equal(res, ref)
        assert res.mode.shape == ref.mode.shape == ()
        res = stats.mode(a, axis=None, keepdims=True, nan_policy='omit')
        ref = stats.mode(a.ravel(), keepdims=True, nan_policy='omit')
        assert_equal(res.mode.ravel(), ref.mode.ravel())
        assert res.mode.shape == (1, 1)
        assert_equal(res.count.ravel(), ref.count.ravel())
        assert res.count.shape == (1, 1)

    @pytest.mark.parametrize('nan_policy', ['propagate', 'omit'])
    def test_gh16955(self, nan_policy):
        shape = (4, 3)
        data = np.ones(shape)
        data[0, 0] = np.nan
        res = stats.mode(a=data, axis=1, keepdims=False, nan_policy=nan_policy)
        assert_array_equal(res.mode, [1, 1, 1, 1])
        assert_array_equal(res.count, [2, 3, 3, 3])
        my_dtype = np.dtype([('asdf', np.uint8), ('qwer', np.float64, (3,))])
        test = np.zeros(10, dtype=my_dtype)
        with pytest.raises(TypeError, match='Argument `a` is not...'):
            stats.mode(test, nan_policy=nan_policy)

    def test_gh9955(self):
        res = stats.mode([])
        ref = (np.nan, 0)
        assert_equal(res, ref)
        res = stats.mode([np.nan], nan_policy='omit')
        assert_equal(res, ref)
        a = [[10.0, 20.0, 20.0], [np.nan, np.nan, np.nan]]
        res = stats.mode(a, axis=1, nan_policy='omit')
        ref = ([20, np.nan], [2, 0])
        assert_equal(res, ref)
        res = stats.mode(a, axis=1, nan_policy='propagate')
        ref = ([20, np.nan], [2, 3])
        assert_equal(res, ref)
        z = np.array([[], []])
        res = stats.mode(z, axis=1)
        ref = ([np.nan, np.nan], [0, 0])
        assert_equal(res, ref)

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    @pytest.mark.parametrize('z', [np.empty((0, 1, 2)), np.empty((1, 1, 2))])
    def test_gh17214(self, z):
        res = stats.mode(z, axis=None, keepdims=True)
        ref = np.mean(z, axis=None, keepdims=True)
        assert res[0].shape == res[1].shape == ref.shape == (1, 1, 1)

    def test_raise_non_numeric_gh18254(self):
        message = 'Argument `a` is not recognized as numeric.'

        class ArrLike:

            def __init__(self, x):
                self._x = x

            def __array__(self):
                return self._x.astype(object)
        with pytest.raises(TypeError, match=message):
            stats.mode(ArrLike(np.arange(3)))
        with pytest.raises(TypeError, match=message):
            stats.mode(np.arange(3, dtype=object))