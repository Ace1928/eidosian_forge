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
def test_trimmed2(self):
    x = [1.2, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0]
    y = [0.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 200.0]
    Xsq1, pval1 = stats.fligner(x, y, center='trimmed', proportiontocut=0.125)
    Xsq2, pval2 = stats.fligner(x[1:-1], y[1:-1], center='mean')
    assert_almost_equal(Xsq1, Xsq2)
    assert_almost_equal(pval1, pval2)