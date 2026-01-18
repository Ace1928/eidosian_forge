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
def test_wilcoxon_tie(self):
    stat, p = stats.wilcoxon([0.1] * 10, mode='approx')
    expected_p = 0.001565402
    assert_equal(stat, 0)
    assert_allclose(p, expected_p, rtol=1e-06)
    stat, p = stats.wilcoxon([0.1] * 10, correction=True, mode='approx')
    expected_p = 0.001904195
    assert_equal(stat, 0)
    assert_allclose(p, expected_p, rtol=1e-06)