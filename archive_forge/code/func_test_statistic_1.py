from itertools import product
import numpy as np
import random
import functools
import pytest
from numpy.testing import (assert_, assert_equal, assert_allclose,
from pytest import raises as assert_raises
import scipy.stats as stats
from scipy.stats import distributions
from scipy.stats._hypotests import (epps_singleton_2samp, cramervonmises,
from scipy.stats._mannwhitneyu import mannwhitneyu, _mwu_state
from .common_tests import check_named_results
from scipy._lib._testutils import _TestPythranFunc
def test_statistic_1(self):
    x = np.array([-0.35, 2.55, 1.73, 0.73, 0.35, 2.69, 0.46, -0.94, -0.37, 12.07])
    y = np.array([-1.15, -0.15, 2.48, 3.25, 3.71, 4.29, 5.0, 7.74, 8.38, 8.6])
    w, p = epps_singleton_2samp(x, y)
    assert_almost_equal(w, 15.14, decimal=1)
    assert_almost_equal(p, 0.00442, decimal=3)