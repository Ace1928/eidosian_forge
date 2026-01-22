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
class TestTtest_1samp:

    def test_vs_nonmasked(self):
        np.random.seed(1234567)
        outcome = np.random.randn(20, 4) + [0, 0, 1, 2]
        res1 = stats.ttest_1samp(outcome[:, 0], 1)
        res2 = mstats.ttest_1samp(outcome[:, 0], 1)
        assert_allclose(res1, res2)

    def test_fully_masked(self):
        np.random.seed(1234567)
        outcome = ma.masked_array(np.random.randn(3), mask=[1, 1, 1])
        expected = (np.nan, np.nan)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'invalid value encountered in absolute')
            for pair in [((np.nan, np.nan), 0.0), (outcome, 0.0)]:
                t, p = mstats.ttest_1samp(*pair)
                assert_array_equal(p, expected)
                assert_array_equal(t, expected)

    def test_result_attributes(self):
        np.random.seed(1234567)
        outcome = np.random.randn(20, 4) + [0, 0, 1, 2]
        res = mstats.ttest_1samp(outcome[:, 0], 1)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes, ma=True)

    def test_empty(self):
        res1 = mstats.ttest_1samp([], 1)
        assert_(np.all(np.isnan(res1)))

    def test_zero_division(self):
        t, p = mstats.ttest_1samp([0, 0, 0], 1)
        assert_equal((np.abs(t), p), (np.inf, 0))
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'invalid value encountered in absolute')
            t, p = mstats.ttest_1samp([0, 0, 0], 0)
            assert_(np.isnan(t))
            assert_array_equal(p, (np.nan, np.nan))

    def test_bad_alternative(self):
        msg = "alternative must be 'less', 'greater' or 'two-sided'"
        with pytest.raises(ValueError, match=msg):
            mstats.ttest_1samp([1, 2, 3], 4, alternative='foo')

    @pytest.mark.parametrize('alternative', ['less', 'greater'])
    def test_alternative(self, alternative):
        x = stats.norm.rvs(loc=10, scale=2, size=100, random_state=123)
        t_ex, p_ex = stats.ttest_1samp(x, 9, alternative=alternative)
        t, p = mstats.ttest_1samp(x, 9, alternative=alternative)
        assert_allclose(t, t_ex, rtol=1e-14)
        assert_allclose(p, p_ex, rtol=1e-14)
        x[1:10] = np.nan
        x = np.ma.masked_array(x, mask=np.isnan(x))
        t_ex, p_ex = stats.ttest_1samp(x.compressed(), 9, alternative=alternative)
        t, p = mstats.ttest_1samp(x, 9, alternative=alternative)
        assert_allclose(t, t_ex, rtol=1e-14)
        assert_allclose(p, p_ex, rtol=1e-14)