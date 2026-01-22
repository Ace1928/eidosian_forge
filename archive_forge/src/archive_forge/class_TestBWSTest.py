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
class TestBWSTest:

    def test_bws_input_validation(self):
        rng = np.random.default_rng(4571775098104213308)
        x, y = rng.random(size=(2, 7))
        message = '`x` and `y` must be exactly one-dimensional.'
        with pytest.raises(ValueError, match=message):
            stats.bws_test([x, x], [y, y])
        message = '`x` and `y` must not contain NaNs.'
        with pytest.raises(ValueError, match=message):
            stats.bws_test([np.nan], y)
        message = '`x` and `y` must be of nonzero size.'
        with pytest.raises(ValueError, match=message):
            stats.bws_test(x, [])
        message = 'alternative` must be one of...'
        with pytest.raises(ValueError, match=message):
            stats.bws_test(x, y, alternative='ekki-ekki')
        message = 'method` must be an instance of...'
        with pytest.raises(ValueError, match=message):
            stats.bws_test(x, y, method=42)

    def test_against_published_reference(self):
        x = [1, 2, 3, 4, 6, 7, 8]
        y = [5, 9, 10, 11, 12, 13, 14]
        res = stats.bws_test(x, y, alternative='two-sided')
        assert_allclose(res.statistic, 5.132, atol=0.001)
        assert_equal(res.pvalue, 10 / 3432)

    @pytest.mark.parametrize(('alternative', 'statistic', 'pvalue'), [('two-sided', 1.7510204081633, 0.1264422777777), ('less', -1.7510204081633, 0.05754662004662), ('greater', -1.7510204081633, 0.9424533799534)])
    def test_against_R(self, alternative, statistic, pvalue):
        rng = np.random.default_rng(4571775098104213308)
        x, y = rng.random(size=(2, 7))
        res = stats.bws_test(x, y, alternative=alternative)
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        assert_allclose(res.pvalue, pvalue, atol=0.01, rtol=0.1)

    @pytest.mark.parametrize(('alternative', 'statistic', 'pvalue'), [('two-sided', 1.142629265891, 0.2903950180801), ('less', 0.99629665877411, 0.8545660222131), ('greater', 0.99629665877411, 0.1454339777869)])
    def test_against_R_imbalanced(self, alternative, statistic, pvalue):
        rng = np.random.default_rng(5429015622386364034)
        x = rng.random(size=9)
        y = rng.random(size=8)
        res = stats.bws_test(x, y, alternative=alternative)
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        assert_allclose(res.pvalue, pvalue, atol=0.01, rtol=0.1)

    def test_method(self):
        rng = np.random.default_rng(1520514347193347862)
        x, y = rng.random(size=(2, 10))
        rng = np.random.default_rng(1520514347193347862)
        method = stats.PermutationMethod(n_resamples=10, random_state=rng)
        res1 = stats.bws_test(x, y, method=method)
        assert len(res1.null_distribution) == 10
        rng = np.random.default_rng(1520514347193347862)
        method = stats.PermutationMethod(n_resamples=10, random_state=rng)
        res2 = stats.bws_test(x, y, method=method)
        assert_allclose(res1.null_distribution, res2.null_distribution)
        rng = np.random.default_rng(5205143471933478621)
        method = stats.PermutationMethod(n_resamples=10, random_state=rng)
        res3 = stats.bws_test(x, y, method=method)
        assert not np.allclose(res3.null_distribution, res1.null_distribution)

    def test_directions(self):
        rng = np.random.default_rng(1520514347193347862)
        x = rng.random(size=5)
        y = x - 1
        res = stats.bws_test(x, y, alternative='greater')
        assert res.statistic > 0
        assert_equal(res.pvalue, 1 / len(res.null_distribution))
        res = stats.bws_test(x, y, alternative='less')
        assert res.statistic > 0
        assert_equal(res.pvalue, 1)
        res = stats.bws_test(y, x, alternative='less')
        assert res.statistic < 0
        assert_equal(res.pvalue, 1 / len(res.null_distribution))
        res = stats.bws_test(y, x, alternative='greater')
        assert res.statistic < 0
        assert_equal(res.pvalue, 1)