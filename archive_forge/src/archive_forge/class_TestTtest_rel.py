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
class TestTtest_rel:

    def test_vs_nonmasked(self):
        np.random.seed(1234567)
        outcome = np.random.randn(20, 4) + [0, 0, 1, 2]
        res1 = stats.ttest_rel(outcome[:, 0], outcome[:, 1])
        res2 = mstats.ttest_rel(outcome[:, 0], outcome[:, 1])
        assert_allclose(res1, res2)
        res1 = stats.ttest_rel(outcome[:, 0], outcome[:, 1], axis=None)
        res2 = mstats.ttest_rel(outcome[:, 0], outcome[:, 1], axis=None)
        assert_allclose(res1, res2)
        res1 = stats.ttest_rel(outcome[:, :2], outcome[:, 2:], axis=0)
        res2 = mstats.ttest_rel(outcome[:, :2], outcome[:, 2:], axis=0)
        assert_allclose(res1, res2)
        res3 = mstats.ttest_rel(outcome[:, :2], outcome[:, 2:])
        assert_allclose(res2, res3)

    def test_fully_masked(self):
        np.random.seed(1234567)
        outcome = ma.masked_array(np.random.randn(3, 2), mask=[[1, 1, 1], [0, 0, 0]])
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'invalid value encountered in absolute')
            for pair in [(outcome[:, 0], outcome[:, 1]), ([np.nan, np.nan], [1.0, 2.0])]:
                t, p = mstats.ttest_rel(*pair)
                assert_array_equal(t, (np.nan, np.nan))
                assert_array_equal(p, (np.nan, np.nan))

    def test_result_attributes(self):
        np.random.seed(1234567)
        outcome = np.random.randn(20, 4) + [0, 0, 1, 2]
        res = mstats.ttest_rel(outcome[:, 0], outcome[:, 1])
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes, ma=True)

    def test_invalid_input_size(self):
        assert_raises(ValueError, mstats.ttest_rel, np.arange(10), np.arange(11))
        x = np.arange(24)
        assert_raises(ValueError, mstats.ttest_rel, x.reshape(2, 3, 4), x.reshape(2, 4, 3), axis=1)
        assert_raises(ValueError, mstats.ttest_rel, x.reshape(2, 3, 4), x.reshape(2, 4, 3), axis=2)

    def test_empty(self):
        res1 = mstats.ttest_rel([], [])
        assert_(np.all(np.isnan(res1)))

    def test_zero_division(self):
        t, p = mstats.ttest_ind([0, 0, 0], [1, 1, 1])
        assert_equal((np.abs(t), p), (np.inf, 0))
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'invalid value encountered in absolute')
            t, p = mstats.ttest_ind([0, 0, 0], [0, 0, 0])
            assert_array_equal(t, np.array([np.nan, np.nan]))
            assert_array_equal(p, np.array([np.nan, np.nan]))

    def test_bad_alternative(self):
        msg = "alternative must be 'less', 'greater' or 'two-sided'"
        with pytest.raises(ValueError, match=msg):
            mstats.ttest_ind([1, 2, 3], [4, 5, 6], alternative='foo')

    @pytest.mark.parametrize('alternative', ['less', 'greater'])
    def test_alternative(self, alternative):
        x = stats.norm.rvs(loc=10, scale=5, size=25, random_state=42)
        y = stats.norm.rvs(loc=8, scale=2, size=25, random_state=42)
        t_ex, p_ex = stats.ttest_rel(x, y, alternative=alternative)
        t, p = mstats.ttest_rel(x, y, alternative=alternative)
        assert_allclose(t, t_ex, rtol=1e-14)
        assert_allclose(p, p_ex, rtol=1e-14)
        x[1:10] = np.nan
        y[1:10] = np.nan
        x = np.ma.masked_array(x, mask=np.isnan(x))
        y = np.ma.masked_array(y, mask=np.isnan(y))
        t, p = mstats.ttest_rel(x, y, alternative=alternative)
        t_ex, p_ex = stats.ttest_rel(x.compressed(), y.compressed(), alternative=alternative)
        assert_allclose(t, t_ex, rtol=1e-14)
        assert_allclose(p, p_ex, rtol=1e-14)