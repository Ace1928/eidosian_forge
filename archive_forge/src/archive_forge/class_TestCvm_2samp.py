from itertools import product
import numpy as np
import random
import functools
import pytest
from numpy.testing import (assert_, assert_equal, assert_allclose,
from pytest import raises as assert_raises
import scipy.stats as stats
from scipy.stats import distributions
from scipy.stats._hypotests import (epps_singleton_2samp, cramervonmises,
from scipy.stats._mannwhitneyu import mannwhitneyu, _mwu_state
from .common_tests import check_named_results
from scipy._lib._testutils import _TestPythranFunc
class TestCvm_2samp:

    def test_invalid_input(self):
        y = np.arange(5)
        msg = 'x and y must contain at least two observations.'
        with pytest.raises(ValueError, match=msg):
            cramervonmises_2samp([], y)
        with pytest.raises(ValueError, match=msg):
            cramervonmises_2samp(y, [1])
        msg = 'method must be either auto, exact or asymptotic'
        with pytest.raises(ValueError, match=msg):
            cramervonmises_2samp(y, y, 'xyz')

    def test_list_input(self):
        x = [2, 3, 4, 7, 6]
        y = [0.2, 0.7, 12, 18]
        r1 = cramervonmises_2samp(x, y)
        r2 = cramervonmises_2samp(np.array(x), np.array(y))
        assert_equal((r1.statistic, r1.pvalue), (r2.statistic, r2.pvalue))

    def test_example_conover(self):
        x = [7.6, 8.4, 8.6, 8.7, 9.3, 9.9, 10.1, 10.6, 11.2]
        y = [5.2, 5.7, 5.9, 6.5, 6.8, 8.2, 9.1, 9.8, 10.8, 11.3, 11.5, 12.3, 12.5, 13.4, 14.6]
        r = cramervonmises_2samp(x, y)
        assert_allclose(r.statistic, 0.262, atol=0.001)
        assert_allclose(r.pvalue, 0.18, atol=0.01)

    @pytest.mark.parametrize('statistic, m, n, pval', [(710, 5, 6, 48.0 / 462), (1897, 7, 7, 117.0 / 1716), (576, 4, 6, 2.0 / 210), (1764, 6, 7, 2.0 / 1716)])
    def test_exact_pvalue(self, statistic, m, n, pval):
        assert_equal(_pval_cvm_2samp_exact(statistic, m, n), pval)

    def test_large_sample(self):
        np.random.seed(4367)
        x = distributions.norm.rvs(size=1000000)
        y = distributions.norm.rvs(size=900000)
        r = cramervonmises_2samp(x, y)
        assert_(0 < r.pvalue < 1)
        r = cramervonmises_2samp(x, y + 0.1)
        assert_(0 < r.pvalue < 1)

    def test_exact_vs_asymptotic(self):
        np.random.seed(0)
        x = np.random.rand(7)
        y = np.random.rand(8)
        r1 = cramervonmises_2samp(x, y, method='exact')
        r2 = cramervonmises_2samp(x, y, method='asymptotic')
        assert_equal(r1.statistic, r2.statistic)
        assert_allclose(r1.pvalue, r2.pvalue, atol=0.01)

    def test_method_auto(self):
        x = np.arange(20)
        y = [0.5, 4.7, 13.1]
        r1 = cramervonmises_2samp(x, y, method='exact')
        r2 = cramervonmises_2samp(x, y, method='auto')
        assert_equal(r1.pvalue, r2.pvalue)
        x = np.arange(21)
        r1 = cramervonmises_2samp(x, y, method='asymptotic')
        r2 = cramervonmises_2samp(x, y, method='auto')
        assert_equal(r1.pvalue, r2.pvalue)

    def test_same_input(self):
        x = np.arange(15)
        res = cramervonmises_2samp(x, x)
        assert_equal((res.statistic, res.pvalue), (0.0, 1.0))
        res = cramervonmises_2samp(x[:4], x[:4])
        assert_equal((res.statistic, res.pvalue), (0.0, 1.0))