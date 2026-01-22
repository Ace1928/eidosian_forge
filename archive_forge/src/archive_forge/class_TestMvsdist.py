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
class TestMvsdist:

    def test_basic(self):
        data = [6, 9, 12, 7, 8, 8, 13]
        mean, var, std = stats.mvsdist(data)
        assert_almost_equal(mean.mean(), 9.0)
        assert_allclose(mean.interval(0.9), (7.103650222612533, 10.896349777387467), rtol=1e-14)
        assert_almost_equal(var.mean(), 10.0)
        assert_allclose(var.interval(0.9), (3.1767242068607087, 24.45910381334018), rtol=1e-09)
        assert_almost_equal(std.mean(), 2.9724954732045084, decimal=14)
        assert_allclose(std.interval(0.9), (1.7823367265645145, 4.945614605014631), rtol=1e-14)

    def test_empty_input(self):
        assert_raises(ValueError, stats.mvsdist, [])

    def test_bad_arg(self):
        data = [1]
        assert_raises(ValueError, stats.mvsdist, data)

    def test_warns(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            [x.mean() for x in stats.mvsdist([1, 2, 3])]
            [x.mean() for x in stats.mvsdist([1, 2, 3, 4, 5])]