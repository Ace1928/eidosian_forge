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
@pytest.mark.parametrize('x', [np.array([2003.0, 1950.0, 1997.0, 2000.0, 2009.0, 2009.0, 1980.0, 1999.0, 2007.0, 1991.0]), np.array([2003.0, 1950.0, 1997.0, 2000.0, 2009.0])])
@pytest.mark.parametrize('scale', [1, 1e-12, 1e-32, 1e-150, 1e+32, 1e+200])
@pytest.mark.parametrize('sign', [1, -1])
def test_overflow_underflow_signed_data(self, x, scale, sign):
    with np.errstate(all='raise'):
        xt_yeo, lam_yeo = stats.yeojohnson(sign * x * scale)
        assert np.all(np.sign(sign * x) == np.sign(xt_yeo))
        assert np.isfinite(lam_yeo)
        assert np.isfinite(np.var(xt_yeo))