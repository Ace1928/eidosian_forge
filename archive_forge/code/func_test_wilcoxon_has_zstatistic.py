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
def test_wilcoxon_has_zstatistic(self):
    rng = np.random.default_rng(89426135444)
    x, y = (rng.random(15), rng.random(15))
    res = stats.wilcoxon(x, y, mode='approx')
    ref = stats.norm.ppf(res.pvalue / 2)
    assert_allclose(res.zstatistic, ref)
    res = stats.wilcoxon(x, y, mode='exact')
    assert not hasattr(res, 'zstatistic')
    res = stats.wilcoxon(x, y)
    assert not hasattr(res, 'zstatistic')