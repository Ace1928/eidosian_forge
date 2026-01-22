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
class TestCvm:

    def test_cdf_4(self):
        assert_allclose(_cdf_cvm([0.02983, 0.04111, 0.12331, 0.94251], 4), [0.01, 0.05, 0.5, 0.999], atol=0.0001)

    def test_cdf_10(self):
        assert_allclose(_cdf_cvm([0.02657, 0.0383, 0.12068, 0.56643], 10), [0.01, 0.05, 0.5, 0.975], atol=0.0001)

    def test_cdf_1000(self):
        assert_allclose(_cdf_cvm([0.02481, 0.03658, 0.11889, 1.1612], 1000), [0.01, 0.05, 0.5, 0.999], atol=0.0001)

    def test_cdf_inf(self):
        assert_allclose(_cdf_cvm([0.0248, 0.03656, 0.11888, 1.16204]), [0.01, 0.05, 0.5, 0.999], atol=0.0001)

    def test_cdf_support(self):
        assert_equal(_cdf_cvm([1 / (12 * 533), 533 / 3], 533), [0, 1])
        assert_equal(_cdf_cvm([1 / (12 * (27 + 1)), (27 + 1) / 3], 27), [0, 1])

    def test_cdf_large_n(self):
        assert_allclose(_cdf_cvm([0.0248, 0.03656, 0.11888, 1.16204, 100], 10000), _cdf_cvm([0.0248, 0.03656, 0.11888, 1.16204, 100]), atol=0.0001)

    def test_large_x(self):
        assert_(0.99999 < _cdf_cvm(333.3, 1000) < 1.0)
        assert_(0.99999 < _cdf_cvm(333.3) < 1.0)

    def test_low_p(self):
        n = 12
        res = cramervonmises(np.ones(n) * 0.8, 'norm')
        assert_(_cdf_cvm(res.statistic, n) > 1.0)
        assert_equal(res.pvalue, 0)

    def test_invalid_input(self):
        assert_raises(ValueError, cramervonmises, [1.5], 'norm')
        assert_raises(ValueError, cramervonmises, (), 'norm')

    def test_values_R(self):
        res = cramervonmises([-1.7, 2, 0, 1.3, 4, 0.1, 0.6], 'norm')
        assert_allclose(res.statistic, 0.288156, atol=1e-06)
        assert_allclose(res.pvalue, 0.1453465, atol=1e-06)
        res = cramervonmises([-1.7, 2, 0, 1.3, 4, 0.1, 0.6], 'norm', (3, 1.5))
        assert_allclose(res.statistic, 0.9426685, atol=1e-06)
        assert_allclose(res.pvalue, 0.002026417, atol=1e-06)
        res = cramervonmises([1, 2, 5, 1.4, 0.14, 11, 13, 0.9, 7.5], 'expon')
        assert_allclose(res.statistic, 0.8421854, atol=1e-06)
        assert_allclose(res.pvalue, 0.004433406, atol=1e-06)

    def test_callable_cdf(self):
        x, args = (np.arange(5), (1.4, 0.7))
        r1 = cramervonmises(x, distributions.expon.cdf)
        r2 = cramervonmises(x, 'expon')
        assert_equal((r1.statistic, r1.pvalue), (r2.statistic, r2.pvalue))
        r1 = cramervonmises(x, distributions.beta.cdf, args)
        r2 = cramervonmises(x, 'beta', args)
        assert_equal((r1.statistic, r1.pvalue), (r2.statistic, r2.pvalue))