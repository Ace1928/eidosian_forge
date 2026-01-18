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
def test_tie_correct(self):
    x = [1, 2, 3, 4]
    y0 = np.array([1, 2, 3, 4, 5])
    dy = np.array([0, 1, 0, 1, 0]) * 0.01
    dy2 = np.array([0, 0, 1, 0, 0]) * 0.01
    y = [y0 - 0.01, y0 - dy, y0 - dy2, y0, y0 + dy2, y0 + dy, y0 + 0.01]
    res = mannwhitneyu(x, y, axis=-1, method='asymptotic')
    U_expected = [10, 9, 8.5, 8, 7.5, 7, 6]
    p_expected = [1, 0.9017048037317, 0.804080657472, 0.7086240584439, 0.6197963884941, 0.5368784563079, 0.3912672792826]
    assert_equal(res.statistic, U_expected)
    assert_allclose(res.pvalue, p_expected)