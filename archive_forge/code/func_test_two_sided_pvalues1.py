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
def test_two_sided_pvalues1(self):
    rtol = 1e-10
    res = stats.binomtest(10079999, 21000000, 0.48)
    assert_allclose(res.pvalue, 1.0, rtol=rtol)
    res = stats.binomtest(10079990, 21000000, 0.48)
    assert_allclose(res.pvalue, 0.9966892187965, rtol=rtol)
    res = stats.binomtest(10080009, 21000000, 0.48)
    assert_allclose(res.pvalue, 0.9970377203856, rtol=rtol)
    res = stats.binomtest(10080017, 21000000, 0.48)
    assert_allclose(res.pvalue, 0.9940754817328, rtol=1e-09)