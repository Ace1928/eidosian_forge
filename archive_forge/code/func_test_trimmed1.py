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
def test_trimmed1(self):
    rs = np.random.RandomState(123)

    def _perturb(g):
        return (np.asarray(g) + 1e-10 * rs.randn(len(g))).tolist()
    g1_ = _perturb(g1)
    g2_ = _perturb(g2)
    g3_ = _perturb(g3)
    Xsq1, pval1 = stats.fligner(g1_, g2_, g3_, center='mean')
    Xsq2, pval2 = stats.fligner(g1_, g2_, g3_, center='trimmed', proportiontocut=0.0)
    assert_almost_equal(Xsq1, Xsq2)
    assert_almost_equal(pval1, pval2)