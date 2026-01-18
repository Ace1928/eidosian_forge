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
def test_somersd_alternative(self):
    x1 = [1, 2, 3, 4, 5]
    x2 = [5, 6, 7, 8, 7]
    expected = stats.somersd(x1, x2, alternative='two-sided')
    assert expected.statistic > 0
    res = stats.somersd(x1, x2, alternative='less')
    assert_equal(res.statistic, expected.statistic)
    assert_allclose(res.pvalue, 1 - expected.pvalue / 2)
    res = stats.somersd(x1, x2, alternative='greater')
    assert_equal(res.statistic, expected.statistic)
    assert_allclose(res.pvalue, expected.pvalue / 2)
    x2.reverse()
    expected = stats.somersd(x1, x2, alternative='two-sided')
    assert expected.statistic < 0
    res = stats.somersd(x1, x2, alternative='greater')
    assert_equal(res.statistic, expected.statistic)
    assert_allclose(res.pvalue, 1 - expected.pvalue / 2)
    res = stats.somersd(x1, x2, alternative='less')
    assert_equal(res.statistic, expected.statistic)
    assert_allclose(res.pvalue, expected.pvalue / 2)
    with pytest.raises(ValueError, match="alternative must be 'less'..."):
        stats.somersd(x1, x2, alternative='ekki-ekki')