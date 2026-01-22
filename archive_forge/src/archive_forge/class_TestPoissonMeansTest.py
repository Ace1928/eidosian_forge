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
class TestPoissonMeansTest:

    @pytest.mark.parametrize('c1, n1, c2, n2, p_expect', ([0, 100, 3, 100, 0.0884], [2, 100, 6, 100, 0.1749]))
    def test_paper_examples(self, c1, n1, c2, n2, p_expect):
        res = stats.poisson_means_test(c1, n1, c2, n2)
        assert_allclose(res.pvalue, p_expect, atol=0.0001)

    @pytest.mark.parametrize('c1, n1, c2, n2, p_expect, alt, d', ([20, 10, 20, 10, 0.999999756892963, 'two-sided', 0], [10, 10, 10, 10, 0.9999998403241203, 'two-sided', 0], [50, 15, 1, 1, 0.09920321053409643, 'two-sided', 0.05], [3, 100, 20, 300, 0.12202725450896404, 'two-sided', 0], [3, 12, 4, 20, 0.40416087318539173, 'greater', 0], [4, 20, 3, 100, 0.008053640402974236, 'greater', 0], [4, 20, 3, 10, 0.3083216325432898, 'less', 0], [1, 1, 50, 15, 0.09322998607245102, 'less', 0]))
    def test_fortran_authors(self, c1, n1, c2, n2, p_expect, alt, d):
        res = stats.poisson_means_test(c1, n1, c2, n2, alternative=alt, diff=d)
        assert_allclose(res.pvalue, p_expect, atol=2e-06, rtol=1e-16)

    def test_different_results(self):
        count1, count2 = (10000, 10000)
        nobs1, nobs2 = (10000, 10000)
        res = stats.poisson_means_test(count1, nobs1, count2, nobs2)
        assert_allclose(res.pvalue, 1)

    def test_less_than_zero_lambda_hat2(self):
        count1, count2 = (0, 0)
        nobs1, nobs2 = (1, 1)
        res = stats.poisson_means_test(count1, nobs1, count2, nobs2)
        assert_allclose(res.pvalue, 1)

    def test_input_validation(self):
        count1, count2 = (0, 0)
        nobs1, nobs2 = (1, 1)
        message = '`k1` and `k2` must be integers.'
        with assert_raises(TypeError, match=message):
            stats.poisson_means_test(0.7, nobs1, count2, nobs2)
        with assert_raises(TypeError, match=message):
            stats.poisson_means_test(count1, nobs1, 0.7, nobs2)
        message = '`k1` and `k2` must be greater than or equal to 0.'
        with assert_raises(ValueError, match=message):
            stats.poisson_means_test(-1, nobs1, count2, nobs2)
        with assert_raises(ValueError, match=message):
            stats.poisson_means_test(count1, nobs1, -1, nobs2)
        message = '`n1` and `n2` must be greater than 0.'
        with assert_raises(ValueError, match=message):
            stats.poisson_means_test(count1, -1, count2, nobs2)
        with assert_raises(ValueError, match=message):
            stats.poisson_means_test(count1, nobs1, count2, -1)
        message = 'diff must be greater than or equal to 0.'
        with assert_raises(ValueError, match=message):
            stats.poisson_means_test(count1, nobs1, count2, nobs2, diff=-1)
        message = 'Alternative must be one of ...'
        with assert_raises(ValueError, match=message):
            stats.poisson_means_test(1, 2, 1, 2, alternative='error')