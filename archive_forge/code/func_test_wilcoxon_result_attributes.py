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
def test_wilcoxon_result_attributes(self):
    x = np.array([120, 114, 181, 188, 180, 146, 121, 191, 132, 113, 127, 112])
    y = np.array([133, 143, 119, 189, 112, 199, 198, 113, 115, 121, 142, 187])
    res = stats.wilcoxon(x, y, correction=False, mode='approx')
    attributes = ('statistic', 'pvalue')
    check_named_results(res, attributes)