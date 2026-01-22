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
class TestIQR:

    def test_basic(self):
        x = np.arange(8) * 0.5
        np.random.shuffle(x)
        assert_equal(stats.iqr(x), 1.75)

    def test_api(self):
        d = np.ones((5, 5))
        stats.iqr(d)
        stats.iqr(d, None)
        stats.iqr(d, 1)
        stats.iqr(d, (0, 1))
        stats.iqr(d, None, (10, 90))
        stats.iqr(d, None, (30, 20), 1.0)
        stats.iqr(d, None, (25, 75), 1.5, 'propagate')
        stats.iqr(d, None, (50, 50), 'normal', 'raise', 'linear')
        stats.iqr(d, None, (25, 75), -0.4, 'omit', 'lower', True)

    def test_empty(self):
        assert_equal(stats.iqr([]), np.nan)
        assert_equal(stats.iqr(np.arange(0)), np.nan)

    def test_constant(self):
        x = np.ones((7, 4))
        assert_equal(stats.iqr(x), 0.0)
        assert_array_equal(stats.iqr(x, axis=0), np.zeros(4))
        assert_array_equal(stats.iqr(x, axis=1), np.zeros(7))
        assert_equal(stats.iqr(x, interpolation='linear'), 0.0)
        assert_equal(stats.iqr(x, interpolation='midpoint'), 0.0)
        assert_equal(stats.iqr(x, interpolation='nearest'), 0.0)
        assert_equal(stats.iqr(x, interpolation='lower'), 0.0)
        assert_equal(stats.iqr(x, interpolation='higher'), 0.0)
        y = np.ones((4, 5, 6)) * np.arange(6)
        assert_array_equal(stats.iqr(y, axis=0), np.zeros((5, 6)))
        assert_array_equal(stats.iqr(y, axis=1), np.zeros((4, 6)))
        assert_array_equal(stats.iqr(y, axis=2), np.full((4, 5), 2.5))
        assert_array_equal(stats.iqr(y, axis=(0, 1)), np.zeros(6))
        assert_array_equal(stats.iqr(y, axis=(0, 2)), np.full(5, 3.0))
        assert_array_equal(stats.iqr(y, axis=(1, 2)), np.full(4, 3.0))

    def test_scalarlike(self):
        x = np.arange(1) + 7.0
        assert_equal(stats.iqr(x[0]), 0.0)
        assert_equal(stats.iqr(x), 0.0)
        assert_array_equal(stats.iqr(x, keepdims=True), [0.0])

    def test_2D(self):
        x = np.arange(15).reshape((3, 5))
        assert_equal(stats.iqr(x), 7.0)
        assert_array_equal(stats.iqr(x, axis=0), np.full(5, 5.0))
        assert_array_equal(stats.iqr(x, axis=1), np.full(3, 2.0))
        assert_array_equal(stats.iqr(x, axis=(0, 1)), 7.0)
        assert_array_equal(stats.iqr(x, axis=(1, 0)), 7.0)

    def test_axis(self):
        o = np.random.normal(size=(71, 23))
        x = np.dstack([o] * 10)
        q = stats.iqr(o)
        assert_equal(stats.iqr(x, axis=(0, 1)), q)
        x = np.moveaxis(x, -1, 0)
        assert_equal(stats.iqr(x, axis=(2, 1)), q)
        x = x.swapaxes(0, 1)
        assert_equal(stats.iqr(x, axis=(0, 2)), q)
        x = x.swapaxes(0, 1)
        assert_equal(stats.iqr(x, axis=(0, 1, 2)), stats.iqr(x, axis=None))
        assert_equal(stats.iqr(x, axis=(0,)), stats.iqr(x, axis=0))
        d = np.arange(3 * 5 * 7 * 11)
        np.random.shuffle(d)
        d = d.reshape((3, 5, 7, 11))
        assert_equal(stats.iqr(d, axis=(0, 1, 2))[0], stats.iqr(d[:, :, :, 0].ravel()))
        assert_equal(stats.iqr(d, axis=(0, 1, 3))[1], stats.iqr(d[:, :, 1, :].ravel()))
        assert_equal(stats.iqr(d, axis=(3, 1, -4))[2], stats.iqr(d[:, :, 2, :].ravel()))
        assert_equal(stats.iqr(d, axis=(3, 1, 2))[2], stats.iqr(d[2, :, :, :].ravel()))
        assert_equal(stats.iqr(d, axis=(3, 2))[2, 1], stats.iqr(d[2, 1, :, :].ravel()))
        assert_equal(stats.iqr(d, axis=(1, -2))[2, 1], stats.iqr(d[2, :, :, 1].ravel()))
        assert_equal(stats.iqr(d, axis=(1, 3))[2, 2], stats.iqr(d[2, :, 2, :].ravel()))
        assert_raises(AxisError, stats.iqr, d, axis=4)
        assert_raises(ValueError, stats.iqr, d, axis=(0, 0))

    def test_rng(self):
        x = np.arange(5)
        assert_equal(stats.iqr(x), 2)
        assert_equal(stats.iqr(x, rng=(25, 87.5)), 2.5)
        assert_equal(stats.iqr(x, rng=(12.5, 75)), 2.5)
        assert_almost_equal(stats.iqr(x, rng=(10, 50)), 1.6)
        assert_raises(ValueError, stats.iqr, x, rng=(0, 101))
        assert_raises(ValueError, stats.iqr, x, rng=(np.nan, 25))
        assert_raises(TypeError, stats.iqr, x, rng=(0, 50, 60))

    def test_interpolation(self):
        x = np.arange(5)
        y = np.arange(4)
        assert_equal(stats.iqr(x), 2)
        assert_equal(stats.iqr(y), 1.5)
        assert_equal(stats.iqr(x, interpolation='linear'), 2)
        assert_equal(stats.iqr(y, interpolation='linear'), 1.5)
        assert_equal(stats.iqr(x, interpolation='higher'), 2)
        assert_equal(stats.iqr(x, rng=(25, 80), interpolation='higher'), 3)
        assert_equal(stats.iqr(y, interpolation='higher'), 2)
        assert_equal(stats.iqr(x, interpolation='lower'), 2)
        assert_equal(stats.iqr(x, rng=(25, 80), interpolation='lower'), 2)
        assert_equal(stats.iqr(y, interpolation='lower'), 2)
        assert_equal(stats.iqr(x, interpolation='nearest'), 2)
        assert_equal(stats.iqr(y, interpolation='nearest'), 1)
        assert_equal(stats.iqr(x, interpolation='midpoint'), 2)
        assert_equal(stats.iqr(x, rng=(25, 80), interpolation='midpoint'), 2.5)
        assert_equal(stats.iqr(y, interpolation='midpoint'), 2)
        for method in ('inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', 'interpolated_inverted_cdf', 'hazen', 'weibull', 'median_unbiased', 'normal_unbiased'):
            stats.iqr(y, interpolation=method)
        assert_raises(ValueError, stats.iqr, x, interpolation='foobar')

    def test_keepdims(self):
        x = np.ones((3, 5, 7, 11))
        assert_equal(stats.iqr(x, axis=None, keepdims=False).shape, ())
        assert_equal(stats.iqr(x, axis=2, keepdims=False).shape, (3, 5, 11))
        assert_equal(stats.iqr(x, axis=(0, 1), keepdims=False).shape, (7, 11))
        assert_equal(stats.iqr(x, axis=(0, 3), keepdims=False).shape, (5, 7))
        assert_equal(stats.iqr(x, axis=(1,), keepdims=False).shape, (3, 7, 11))
        assert_equal(stats.iqr(x, (0, 1, 2, 3), keepdims=False).shape, ())
        assert_equal(stats.iqr(x, axis=(0, 1, 3), keepdims=False).shape, (7,))
        assert_equal(stats.iqr(x, axis=None, keepdims=True).shape, (1, 1, 1, 1))
        assert_equal(stats.iqr(x, axis=2, keepdims=True).shape, (3, 5, 1, 11))
        assert_equal(stats.iqr(x, axis=(0, 1), keepdims=True).shape, (1, 1, 7, 11))
        assert_equal(stats.iqr(x, axis=(0, 3), keepdims=True).shape, (1, 5, 7, 1))
        assert_equal(stats.iqr(x, axis=(1,), keepdims=True).shape, (3, 1, 7, 11))
        assert_equal(stats.iqr(x, (0, 1, 2, 3), keepdims=True).shape, (1, 1, 1, 1))
        assert_equal(stats.iqr(x, axis=(0, 1, 3), keepdims=True).shape, (1, 1, 7, 1))

    def test_nanpolicy(self):
        x = np.arange(15.0).reshape((3, 5))
        assert_equal(stats.iqr(x, nan_policy='propagate'), 7)
        assert_equal(stats.iqr(x, nan_policy='omit'), 7)
        assert_equal(stats.iqr(x, nan_policy='raise'), 7)
        x[1, 2] = np.nan
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            assert_equal(stats.iqr(x, nan_policy='propagate'), np.nan)
            assert_equal(stats.iqr(x, axis=0, nan_policy='propagate'), [5, 5, np.nan, 5, 5])
            assert_equal(stats.iqr(x, axis=1, nan_policy='propagate'), [2, np.nan, 2])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            assert_equal(stats.iqr(x, nan_policy='omit'), 7.5)
            assert_equal(stats.iqr(x, axis=0, nan_policy='omit'), np.full(5, 5))
            assert_equal(stats.iqr(x, axis=1, nan_policy='omit'), [2, 2.5, 2])
        assert_raises(ValueError, stats.iqr, x, nan_policy='raise')
        assert_raises(ValueError, stats.iqr, x, axis=0, nan_policy='raise')
        assert_raises(ValueError, stats.iqr, x, axis=1, nan_policy='raise')
        assert_raises(ValueError, stats.iqr, x, nan_policy='barfood')

    def test_scale(self):
        x = np.arange(15.0).reshape((3, 5))
        assert_equal(stats.iqr(x, scale=1.0), 7)
        assert_almost_equal(stats.iqr(x, scale='normal'), 7 / 1.3489795)
        assert_equal(stats.iqr(x, scale=2.0), 3.5)
        x[1, 2] = np.nan
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            assert_equal(stats.iqr(x, scale=1.0, nan_policy='propagate'), np.nan)
            assert_equal(stats.iqr(x, scale='normal', nan_policy='propagate'), np.nan)
            assert_equal(stats.iqr(x, scale=2.0, nan_policy='propagate'), np.nan)
            assert_equal(stats.iqr(x, axis=1, scale=1.0, nan_policy='propagate'), [2, np.nan, 2])
            assert_almost_equal(stats.iqr(x, axis=1, scale='normal', nan_policy='propagate'), np.array([2, np.nan, 2]) / 1.3489795)
            assert_equal(stats.iqr(x, axis=1, scale=2.0, nan_policy='propagate'), [1, np.nan, 1])
        assert_equal(stats.iqr(x, scale=1.0, nan_policy='omit'), 7.5)
        assert_almost_equal(stats.iqr(x, scale='normal', nan_policy='omit'), 7.5 / 1.3489795)
        assert_equal(stats.iqr(x, scale=2.0, nan_policy='omit'), 3.75)
        assert_raises(ValueError, stats.iqr, x, scale='foobar')