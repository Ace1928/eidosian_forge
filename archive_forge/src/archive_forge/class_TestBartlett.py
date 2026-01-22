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
class TestBartlett:

    def test_data(self):
        args = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]
        T, pval = stats.bartlett(*args)
        assert_almost_equal(T, 20.78587342806484, 7)
        assert_almost_equal(pval, 0.0136358632781, 7)

    def test_bad_arg(self):
        assert_raises(ValueError, stats.bartlett, [1])

    def test_result_attributes(self):
        args = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]
        res = stats.bartlett(*args)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes)

    def test_empty_arg(self):
        args = (g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, [])
        assert_equal((np.nan, np.nan), stats.bartlett(*args))

    def test_1d_input(self):
        x = np.array([[1, 2], [3, 4]])
        assert_raises(ValueError, stats.bartlett, g1, x)