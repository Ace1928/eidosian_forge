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
def test_ties_options(self):
    x = [1, 2, 3, 4]
    y = [5, 6]
    z = [7, 8, 9]
    stat, p, m, tbl = stats.median_test(x, y, z)
    assert_equal(m, 5)
    assert_equal(tbl, [[0, 1, 3], [4, 1, 0]])
    stat, p, m, tbl = stats.median_test(x, y, z, ties='ignore')
    assert_equal(m, 5)
    assert_equal(tbl, [[0, 1, 3], [4, 0, 0]])
    stat, p, m, tbl = stats.median_test(x, y, z, ties='above')
    assert_equal(m, 5)
    assert_equal(tbl, [[0, 2, 3], [4, 0, 0]])