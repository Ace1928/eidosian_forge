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
class TestCorrSpearmanr:
    """ W.II.D. Compute a correlation matrix on all the variables.

        All the correlations, except for ZERO and MISS, should be exactly 1.
        ZERO and MISS should have undefined or missing correlations with the
        other variables.  The same should go for SPEARMAN correlations, if
        your program has them.
    """

    def test_scalar(self):
        y = stats.spearmanr(4.0, 2.0)
        assert_(np.isnan(y).all())

    def test_uneven_lengths(self):
        assert_raises(ValueError, stats.spearmanr, [1, 2, 1], [8, 9])
        assert_raises(ValueError, stats.spearmanr, [1, 2, 1], 8)

    def test_uneven_2d_shapes(self):
        np.random.seed(232324)
        x = np.random.randn(4, 3)
        y = np.random.randn(4, 2)
        assert stats.spearmanr(x, y).statistic.shape == (5, 5)
        assert stats.spearmanr(x.T, y.T, axis=1).pvalue.shape == (5, 5)
        assert_raises(ValueError, stats.spearmanr, x, y, axis=1)
        assert_raises(ValueError, stats.spearmanr, x.T, y.T)

    def test_ndim_too_high(self):
        np.random.seed(232324)
        x = np.random.randn(4, 3, 2)
        assert_raises(ValueError, stats.spearmanr, x)
        assert_raises(ValueError, stats.spearmanr, x, x)
        assert_raises(ValueError, stats.spearmanr, x, None, None)
        assert_allclose(stats.spearmanr(x, x, axis=None), stats.spearmanr(x.flatten(), x.flatten(), axis=0))

    def test_nan_policy(self):
        x = np.arange(10.0)
        x[9] = np.nan
        assert_array_equal(stats.spearmanr(x, x), (np.nan, np.nan))
        assert_array_equal(stats.spearmanr(x, x, nan_policy='omit'), (1.0, 0.0))
        assert_raises(ValueError, stats.spearmanr, x, x, nan_policy='raise')
        assert_raises(ValueError, stats.spearmanr, x, x, nan_policy='foobar')

    def test_nan_policy_bug_12458(self):
        np.random.seed(5)
        x = np.random.rand(5, 10)
        k = 6
        x[:, k] = np.nan
        y = np.delete(x, k, axis=1)
        corx, px = stats.spearmanr(x, nan_policy='omit')
        cory, py = stats.spearmanr(y)
        corx = np.delete(np.delete(corx, k, axis=1), k, axis=0)
        px = np.delete(np.delete(px, k, axis=1), k, axis=0)
        assert_allclose(corx, cory, atol=1e-14)
        assert_allclose(px, py, atol=1e-14)

    def test_nan_policy_bug_12411(self):
        np.random.seed(5)
        m = 5
        n = 10
        x = np.random.randn(m, n)
        x[1, 0] = np.nan
        x[3, -1] = np.nan
        corr, pvalue = stats.spearmanr(x, axis=1, nan_policy='propagate')
        res = [[stats.spearmanr(x[i, :], x[j, :]).statistic for i in range(m)] for j in range(m)]
        assert_allclose(corr, res)

    def test_sXX(self):
        y = stats.spearmanr(X, X)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_sXBIG(self):
        y = stats.spearmanr(X, BIG)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_sXLITTLE(self):
        y = stats.spearmanr(X, LITTLE)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_sXHUGE(self):
        y = stats.spearmanr(X, HUGE)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_sXTINY(self):
        y = stats.spearmanr(X, TINY)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_sXROUND(self):
        y = stats.spearmanr(X, ROUND)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_sBIGBIG(self):
        y = stats.spearmanr(BIG, BIG)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_sBIGLITTLE(self):
        y = stats.spearmanr(BIG, LITTLE)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_sBIGHUGE(self):
        y = stats.spearmanr(BIG, HUGE)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_sBIGTINY(self):
        y = stats.spearmanr(BIG, TINY)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_sBIGROUND(self):
        y = stats.spearmanr(BIG, ROUND)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_sLITTLELITTLE(self):
        y = stats.spearmanr(LITTLE, LITTLE)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_sLITTLEHUGE(self):
        y = stats.spearmanr(LITTLE, HUGE)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_sLITTLETINY(self):
        y = stats.spearmanr(LITTLE, TINY)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_sLITTLEROUND(self):
        y = stats.spearmanr(LITTLE, ROUND)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_sHUGEHUGE(self):
        y = stats.spearmanr(HUGE, HUGE)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_sHUGETINY(self):
        y = stats.spearmanr(HUGE, TINY)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_sHUGEROUND(self):
        y = stats.spearmanr(HUGE, ROUND)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_sTINYTINY(self):
        y = stats.spearmanr(TINY, TINY)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_sTINYROUND(self):
        y = stats.spearmanr(TINY, ROUND)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_sROUNDROUND(self):
        y = stats.spearmanr(ROUND, ROUND)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_spearmanr_result_attributes(self):
        res = stats.spearmanr(X, X)
        attributes = ('correlation', 'pvalue')
        check_named_results(res, attributes)
        assert_equal(res.correlation, res.statistic)

    def test_1d_vs_2d(self):
        x1 = [1, 2, 3, 4, 5, 6]
        x2 = [1, 2, 3, 4, 6, 5]
        res1 = stats.spearmanr(x1, x2)
        res2 = stats.spearmanr(np.asarray([x1, x2]).T)
        assert_allclose(res1, res2)

    def test_1d_vs_2d_nans(self):
        for nan_policy in ['propagate', 'omit']:
            x1 = [1, np.nan, 3, 4, 5, 6]
            x2 = [1, 2, 3, 4, 6, np.nan]
            res1 = stats.spearmanr(x1, x2, nan_policy=nan_policy)
            res2 = stats.spearmanr(np.asarray([x1, x2]).T, nan_policy=nan_policy)
            assert_allclose(res1, res2)

    def test_3cols(self):
        x1 = np.arange(6)
        x2 = -x1
        x3 = np.array([0, 1, 2, 3, 5, 4])
        x = np.asarray([x1, x2, x3]).T
        actual = stats.spearmanr(x)
        expected_corr = np.array([[1, -1, 0.94285714], [-1, 1, -0.94285714], [0.94285714, -0.94285714, 1]])
        expected_pvalue = np.zeros((3, 3), dtype=float)
        expected_pvalue[2, 0:2] = 0.00480466472
        expected_pvalue[0:2, 2] = 0.00480466472
        assert_allclose(actual.statistic, expected_corr)
        assert_allclose(actual.pvalue, expected_pvalue)

    def test_gh_9103(self):
        x = np.array([[np.nan, 3.0, 4.0, 5.0, 5.1, 6.0, 9.2], [5.0, np.nan, 4.1, 4.8, 4.9, 5.0, 4.1], [0.5, 4.0, 7.1, 3.8, 8.0, 5.1, 7.6]]).T
        corr = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, 1.0]])
        assert_allclose(stats.spearmanr(x, nan_policy='propagate').statistic, corr)
        res = stats.spearmanr(x, nan_policy='omit').statistic
        assert_allclose((res[0][1], res[0][2], res[1][2]), (0.2051957, 0.4857143, -0.4707919), rtol=1e-06)

    def test_gh_8111(self):
        n = 100
        np.random.seed(234568)
        x = np.random.rand(n)
        m = np.random.rand(n) > 0.7
        a = x > 0.5
        b = np.array(x)
        res1 = stats.spearmanr(a, b, nan_policy='omit').statistic
        b[m] = np.nan
        res2 = stats.spearmanr(a, b, nan_policy='omit').statistic
        a = a.astype(np.int32)
        res3 = stats.spearmanr(a, b, nan_policy='omit').statistic
        expected = [0.865895477, 0.866100381, 0.866100381]
        assert_allclose([res1, res2, res3], expected)