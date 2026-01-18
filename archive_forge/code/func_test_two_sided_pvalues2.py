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
def test_two_sided_pvalues2(self):
    rtol = 1e-10
    res = stats.binomtest(9, n=21, p=0.48)
    assert_allclose(res.pvalue, 0.6689672431939, rtol=rtol)
    res = stats.binomtest(4, 21, 0.48)
    assert_allclose(res.pvalue, 0.008139563452106, rtol=rtol)
    res = stats.binomtest(11, 21, 0.48)
    assert_allclose(res.pvalue, 0.8278629664608, rtol=rtol)
    res = stats.binomtest(7, 21, 0.48)
    assert_allclose(res.pvalue, 0.1966772901718, rtol=rtol)
    res = stats.binomtest(3, 10, 0.5)
    assert_allclose(res.pvalue, 0.34375, rtol=rtol)
    res = stats.binomtest(2, 2, 0.4)
    assert_allclose(res.pvalue, 0.16, rtol=rtol)
    res = stats.binomtest(2, 4, 0.3)
    assert_allclose(res.pvalue, 0.5884, rtol=rtol)