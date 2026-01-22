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
class TestCorrPearsonr:
    """ W.II.D. Compute a correlation matrix on all the variables.

        All the correlations, except for ZERO and MISS, should be exactly 1.
        ZERO and MISS should have undefined or missing correlations with the
        other variables.  The same should go for SPEARMAN correlations, if
        your program has them.
    """

    def test_pXX(self):
        y = stats.pearsonr(X, X)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_pXBIG(self):
        y = stats.pearsonr(X, BIG)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_pXLITTLE(self):
        y = stats.pearsonr(X, LITTLE)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_pXHUGE(self):
        y = stats.pearsonr(X, HUGE)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_pXTINY(self):
        y = stats.pearsonr(X, TINY)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_pXROUND(self):
        y = stats.pearsonr(X, ROUND)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_pBIGBIG(self):
        y = stats.pearsonr(BIG, BIG)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_pBIGLITTLE(self):
        y = stats.pearsonr(BIG, LITTLE)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_pBIGHUGE(self):
        y = stats.pearsonr(BIG, HUGE)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_pBIGTINY(self):
        y = stats.pearsonr(BIG, TINY)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_pBIGROUND(self):
        y = stats.pearsonr(BIG, ROUND)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_pLITTLELITTLE(self):
        y = stats.pearsonr(LITTLE, LITTLE)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_pLITTLEHUGE(self):
        y = stats.pearsonr(LITTLE, HUGE)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_pLITTLETINY(self):
        y = stats.pearsonr(LITTLE, TINY)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_pLITTLEROUND(self):
        y = stats.pearsonr(LITTLE, ROUND)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_pHUGEHUGE(self):
        y = stats.pearsonr(HUGE, HUGE)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_pHUGETINY(self):
        y = stats.pearsonr(HUGE, TINY)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_pHUGEROUND(self):
        y = stats.pearsonr(HUGE, ROUND)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_pTINYTINY(self):
        y = stats.pearsonr(TINY, TINY)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_pTINYROUND(self):
        y = stats.pearsonr(TINY, ROUND)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_pROUNDROUND(self):
        y = stats.pearsonr(ROUND, ROUND)
        r = y[0]
        assert_approx_equal(r, 1.0)

    def test_pearsonr_result_attributes(self):
        res = stats.pearsonr(X, X)
        attributes = ('correlation', 'pvalue')
        check_named_results(res, attributes)
        assert_equal(res.correlation, res.statistic)

    def test_r_almost_exactly_pos1(self):
        a = arange(3.0)
        r, prob = stats.pearsonr(a, a)
        assert_allclose(r, 1.0, atol=1e-15)
        assert_allclose(prob, 0.0, atol=np.sqrt(2 * np.spacing(1.0)))

    def test_r_almost_exactly_neg1(self):
        a = arange(3.0)
        r, prob = stats.pearsonr(a, -a)
        assert_allclose(r, -1.0, atol=1e-15)
        assert_allclose(prob, 0.0, atol=np.sqrt(2 * np.spacing(1.0)))

    def test_basic(self):
        a = array([-1, 0, 1])
        b = array([0, 0, 3])
        r, prob = stats.pearsonr(a, b)
        assert_approx_equal(r, np.sqrt(3) / 2)
        assert_approx_equal(prob, 1 / 3)

    def test_constant_input(self):
        msg = 'An input array is constant'
        with assert_warns(stats.ConstantInputWarning, match=msg):
            r, p = stats.pearsonr([0.667, 0.667, 0.667], [0.123, 0.456, 0.789])
            assert_equal(r, np.nan)
            assert_equal(p, np.nan)

    def test_near_constant_input(self):
        x = [2, 2, 2 + np.spacing(2)]
        y = [3, 3, 3 + 6 * np.spacing(3)]
        msg = 'An input array is nearly constant; the computed'
        with assert_warns(stats.NearConstantInputWarning, match=msg):
            r, p = stats.pearsonr(x, y)

    def test_very_small_input_values(self):
        x = [0.004434375, 0.004756007, 0.003911996, 0.0038005, 0.003409971]
        y = [2.48e-188, 7.41e-181, 4.09e-208, 2.08e-223, 2.66e-245]
        r, p = stats.pearsonr(x, y)
        assert_allclose(r, 0.727293054075045)
        assert_allclose(p, 0.1637805429533202)

    def test_very_large_input_values(self):
        x = 1e+90 * np.array([0, 0, 0, 1, 1, 1, 1])
        y = 1e+90 * np.arange(7)
        r, p = stats.pearsonr(x, y)
        assert_allclose(r, 0.8660254037844386)
        assert_allclose(p, 0.011724811003954639)

    def test_extremely_large_input_values(self):
        x = np.array([2.3e+200, 4.5e+200, 6.7e+200, 8e+200])
        y = np.array([1.2e+199, 5.5e+200, 3.3e+201, 1e+200])
        r, p = stats.pearsonr(x, y)
        assert_allclose(r, 0.351312332103289)
        assert_allclose(p, 0.648687667896711)

    def test_length_two_pos1(self):
        res = stats.pearsonr([1, 2], [3, 5])
        r, p = res
        assert_equal(r, 1)
        assert_equal(p, 1)
        assert_equal(res.confidence_interval(), (-1, 1))

    def test_length_two_neg2(self):
        r, p = stats.pearsonr([2, 1], [3, 5])
        assert_equal(r, -1)
        assert_equal(p, 1)

    @pytest.mark.parametrize('alternative, pval, rlow, rhigh, sign', [('two-sided', 0.325800137536, -0.814938968841, 0.99230697523, 1), ('less', 0.8370999312316, -1, 0.985600937290653, 1), ('greater', 0.1629000687684, -0.6785654158217636, 1, 1), ('two-sided', 0.325800137536, -0.992306975236, 0.81493896884, -1), ('less', 0.1629000687684, -1.0, 0.6785654158217636, -1), ('greater', 0.8370999312316, -0.985600937290653, 1.0, -1)])
    def test_basic_example(self, alternative, pval, rlow, rhigh, sign):
        x = [1, 2, 3, 4]
        y = np.array([0, 1, 0.5, 1]) * sign
        result = stats.pearsonr(x, y, alternative=alternative)
        assert_allclose(result.statistic, 0.6741998624632421 * sign, rtol=1e-12)
        assert_allclose(result.pvalue, pval, rtol=1e-06)
        ci = result.confidence_interval()
        assert_allclose(ci, (rlow, rhigh), rtol=1e-06)

    def test_negative_correlation_pvalue_gh17795(self):
        x = np.arange(10)
        y = -x
        test_greater = stats.pearsonr(x, y, alternative='greater')
        test_less = stats.pearsonr(x, y, alternative='less')
        assert_allclose(test_greater.pvalue, 1)
        assert_allclose(test_less.pvalue, 0, atol=1e-20)

    def test_length3_r_exactly_negative_one(self):
        x = [1, 2, 3]
        y = [5, -4, -13]
        res = stats.pearsonr(x, y)
        r, p = res
        assert_allclose(r, -1.0)
        assert_allclose(p, 0.0, atol=1e-07)
        assert_equal(res.confidence_interval(), (-1, 1))

    def test_unequal_lengths(self):
        x = [1, 2, 3]
        y = [4, 5]
        assert_raises(ValueError, stats.pearsonr, x, y)

    def test_len1(self):
        x = [1]
        y = [2]
        assert_raises(ValueError, stats.pearsonr, x, y)

    def test_complex_data(self):
        x = [-1j, -2j, -3j]
        y = [-1j, -2j, -3j]
        message = 'This function does not support complex data'
        with pytest.raises(ValueError, match=message):
            stats.pearsonr(x, y)

    @pytest.mark.xslow
    @pytest.mark.parametrize('alternative', ('less', 'greater', 'two-sided'))
    @pytest.mark.parametrize('method', ('permutation', 'monte_carlo'))
    def test_resampling_pvalue(self, method, alternative):
        rng = np.random.default_rng(24623935790378923)
        size = 100 if method == 'permutation' else 1000
        x = rng.normal(size=size)
        y = rng.normal(size=size)
        methods = {'permutation': stats.PermutationMethod(random_state=rng), 'monte_carlo': stats.MonteCarloMethod(rvs=(rng.normal,) * 2)}
        method = methods[method]
        res = stats.pearsonr(x, y, alternative=alternative, method=method)
        ref = stats.pearsonr(x, y, alternative=alternative)
        assert_allclose(res.statistic, ref.statistic, rtol=1e-15)
        assert_allclose(res.pvalue, ref.pvalue, rtol=0.01, atol=0.001)

    @pytest.mark.xslow
    @pytest.mark.parametrize('alternative', ('less', 'greater', 'two-sided'))
    def test_bootstrap_ci(self, alternative):
        rng = np.random.default_rng(24623935790378923)
        x = rng.normal(size=100)
        y = rng.normal(size=100)
        res = stats.pearsonr(x, y, alternative=alternative)
        method = stats.BootstrapMethod(random_state=rng)
        res_ci = res.confidence_interval(method=method)
        ref_ci = res.confidence_interval()
        assert_allclose(res_ci, ref_ci, atol=0.01)

    def test_invalid_method(self):
        message = '`method` must be an instance of...'
        with pytest.raises(ValueError, match=message):
            stats.pearsonr([1, 2], [3, 4], method='asymptotic')
        res = stats.pearsonr([1, 2], [3, 4])
        with pytest.raises(ValueError, match=message):
            res.confidence_interval(method='exact')