import warnings
import platform
import numpy as np
from numpy import nan
import numpy.ma as ma
from numpy.ma import masked, nomask
import scipy.stats.mstats as mstats
from scipy import stats
from .common_tests import check_named_results
import pytest
from pytest import raises as assert_raises
from numpy.ma.testutils import (assert_equal, assert_almost_equal,
from numpy.testing import suppress_warnings
from scipy.stats import _mstats_basic
class TestTrimming:

    def test_trim(self):
        a = ma.arange(10)
        assert_equal(mstats.trim(a), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        a = ma.arange(10)
        assert_equal(mstats.trim(a, (2, 8)), [None, None, 2, 3, 4, 5, 6, 7, 8, None])
        a = ma.arange(10)
        assert_equal(mstats.trim(a, limits=(2, 8), inclusive=(False, False)), [None, None, None, 3, 4, 5, 6, 7, None, None])
        a = ma.arange(10)
        assert_equal(mstats.trim(a, limits=(0.1, 0.2), relative=True), [None, 1, 2, 3, 4, 5, 6, 7, None, None])
        a = ma.arange(12)
        a[[0, -1]] = a[5] = masked
        assert_equal(mstats.trim(a, (2, 8)), [None, None, 2, 3, 4, None, 6, 7, 8, None, None, None])
        x = ma.arange(100).reshape(10, 10)
        expected = [1] * 10 + [0] * 70 + [1] * 20
        trimx = mstats.trim(x, (0.1, 0.2), relative=True, axis=None)
        assert_equal(trimx._mask.ravel(), expected)
        trimx = mstats.trim(x, (0.1, 0.2), relative=True, axis=0)
        assert_equal(trimx._mask.ravel(), expected)
        trimx = mstats.trim(x, (0.1, 0.2), relative=True, axis=-1)
        assert_equal(trimx._mask.T.ravel(), expected)
        x = ma.arange(110).reshape(11, 10)
        x[1] = masked
        expected = [1] * 20 + [0] * 70 + [1] * 20
        trimx = mstats.trim(x, (0.1, 0.2), relative=True, axis=None)
        assert_equal(trimx._mask.ravel(), expected)
        trimx = mstats.trim(x, (0.1, 0.2), relative=True, axis=0)
        assert_equal(trimx._mask.ravel(), expected)
        trimx = mstats.trim(x.T, (0.1, 0.2), relative=True, axis=-1)
        assert_equal(trimx.T._mask.ravel(), expected)

    def test_trim_old(self):
        x = ma.arange(100)
        assert_equal(mstats.trimboth(x).count(), 60)
        assert_equal(mstats.trimtail(x, tail='r').count(), 80)
        x[50:70] = masked
        trimx = mstats.trimboth(x)
        assert_equal(trimx.count(), 48)
        assert_equal(trimx._mask, [1] * 16 + [0] * 34 + [1] * 20 + [0] * 14 + [1] * 16)
        x._mask = nomask
        x.shape = (10, 10)
        assert_equal(mstats.trimboth(x).count(), 60)
        assert_equal(mstats.trimtail(x).count(), 80)

    def test_trimr(self):
        x = ma.arange(10)
        result = mstats.trimr(x, limits=(0.15, 0.14), inclusive=(False, False))
        expected = ma.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], mask=[1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
        assert_equal(result, expected)
        assert_equal(result.mask, expected.mask)

    def test_trimmedmean(self):
        data = ma.array([77, 87, 88, 114, 151, 210, 219, 246, 253, 262, 296, 299, 306, 376, 428, 515, 666, 1310, 2611])
        assert_almost_equal(mstats.trimmed_mean(data, 0.1), 343, 0)
        assert_almost_equal(mstats.trimmed_mean(data, (0.1, 0.1)), 343, 0)
        assert_almost_equal(mstats.trimmed_mean(data, (0.2, 0.2)), 283, 0)

    def test_trimmedvar(self):
        rng = np.random.default_rng(3262323289434724460)
        data_orig = rng.random(size=20)
        data = np.sort(data_orig)
        data = ma.array(data, mask=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        assert_allclose(mstats.trimmed_var(data_orig, 0.1), data.var())

    def test_trimmedstd(self):
        rng = np.random.default_rng(7121029245207162780)
        data_orig = rng.random(size=20)
        data = np.sort(data_orig)
        data = ma.array(data, mask=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        assert_allclose(mstats.trimmed_std(data_orig, 0.1), data.std())

    def test_trimmed_stde(self):
        data = ma.array([77, 87, 88, 114, 151, 210, 219, 246, 253, 262, 296, 299, 306, 376, 428, 515, 666, 1310, 2611])
        assert_almost_equal(mstats.trimmed_stde(data, (0.2, 0.2)), 56.13193, 5)
        assert_almost_equal(mstats.trimmed_stde(data, 0.2), 56.13193, 5)

    def test_winsorization(self):
        data = ma.array([77, 87, 88, 114, 151, 210, 219, 246, 253, 262, 296, 299, 306, 376, 428, 515, 666, 1310, 2611])
        assert_almost_equal(mstats.winsorize(data, (0.2, 0.2)).var(ddof=1), 21551.4, 1)
        assert_almost_equal(mstats.winsorize(data, (0.2, 0.2), (False, False)).var(ddof=1), 11887.3, 1)
        data[5] = masked
        winsorized = mstats.winsorize(data)
        assert_equal(winsorized.mask, data.mask)

    def test_winsorization_nan(self):
        data = ma.array([np.nan, np.nan, 0, 1, 2])
        assert_raises(ValueError, mstats.winsorize, data, (0.05, 0.05), nan_policy='raise')
        assert_equal(mstats.winsorize(data, (0.4, 0.4)), ma.array([2, 2, 2, 2, 2]))
        assert_equal(mstats.winsorize(data, (0.8, 0.8)), ma.array([np.nan, np.nan, np.nan, np.nan, np.nan]))
        assert_equal(mstats.winsorize(data, (0.4, 0.4), nan_policy='omit'), ma.array([np.nan, np.nan, 2, 2, 2]))
        assert_equal(mstats.winsorize(data, (0.8, 0.8), nan_policy='omit'), ma.array([np.nan, np.nan, 2, 2, 2]))