import warnings
import sys
from functools import partial
import numpy as np
from numpy.random import RandomState
from numpy.testing import (assert_array_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import re
from scipy import optimize, stats, special
from scipy.stats._morestats import _abw_state, _get_As_weibull, _Avals_weibull
from .common_tests import check_named_results
from .._hypotests import _get_wilcoxon_distr, _get_wilcoxon_distr2
from scipy.stats._binomtest import _binary_search_for_binom_tst
from scipy.stats._distr_params import distcont
class TestLevene:

    def test_data(self):
        args = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]
        W, pval = stats.levene(*args)
        assert_almost_equal(W, 1.705917693000894, 7)
        assert_almost_equal(pval, 0.0990829755522, 7)

    def test_trimmed1(self):
        W1, pval1 = stats.levene(g1, g2, g3, center='mean')
        W2, pval2 = stats.levene(g1, g2, g3, center='trimmed', proportiontocut=0.0)
        assert_almost_equal(W1, W2)
        assert_almost_equal(pval1, pval2)

    def test_trimmed2(self):
        x = [1.2, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0]
        y = [0.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 200.0]
        np.random.seed(1234)
        x2 = np.random.permutation(x)
        W0, pval0 = stats.levene(x, y, center='trimmed', proportiontocut=0.125)
        W1, pval1 = stats.levene(x2, y, center='trimmed', proportiontocut=0.125)
        W2, pval2 = stats.levene(x[1:-1], y[1:-1], center='mean')
        assert_almost_equal(W0, W2)
        assert_almost_equal(W1, W2)
        assert_almost_equal(pval1, pval2)

    def test_equal_mean_median(self):
        x = np.linspace(-1, 1, 21)
        np.random.seed(1234)
        x2 = np.random.permutation(x)
        y = x ** 3
        W1, pval1 = stats.levene(x, y, center='mean')
        W2, pval2 = stats.levene(x2, y, center='median')
        assert_almost_equal(W1, W2)
        assert_almost_equal(pval1, pval2)

    def test_bad_keyword(self):
        x = np.linspace(-1, 1, 21)
        assert_raises(TypeError, stats.levene, x, x, portiontocut=0.1)

    def test_bad_center_value(self):
        x = np.linspace(-1, 1, 21)
        assert_raises(ValueError, stats.levene, x, x, center='trim')

    def test_too_few_args(self):
        assert_raises(ValueError, stats.levene, [1])

    def test_result_attributes(self):
        args = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]
        res = stats.levene(*args)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes)

    def test_1d_input(self):
        x = np.array([[1, 2], [3, 4]])
        assert_raises(ValueError, stats.levene, g1, x)