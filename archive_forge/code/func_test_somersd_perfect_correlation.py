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
@pytest.mark.parametrize('positive_correlation', (False, True))
def test_somersd_perfect_correlation(self, positive_correlation):
    x1 = np.arange(10)
    x2 = x1 if positive_correlation else np.flip(x1)
    expected_statistic = 1 if positive_correlation else -1
    res = stats.somersd(x1, x2, alternative='two-sided')
    assert res.statistic == expected_statistic
    assert res.pvalue == 0
    res = stats.somersd(x1, x2, alternative='less')
    assert res.statistic == expected_statistic
    assert res.pvalue == (1 if positive_correlation else 0)
    res = stats.somersd(x1, x2, alternative='greater')
    assert res.statistic == expected_statistic
    assert res.pvalue == (0 if positive_correlation else 1)