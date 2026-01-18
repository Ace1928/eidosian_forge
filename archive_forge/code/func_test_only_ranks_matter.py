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
def test_only_ranks_matter(self):
    x = [1, 2, 3]
    x2 = [-1, 2.1, np.inf]
    y = [3, 2, 1]
    y2 = [0, -0.5, -np.inf]
    res = stats.somersd(x, y)
    res2 = stats.somersd(x2, y2)
    assert_equal(res.statistic, res2.statistic)
    assert_equal(res.pvalue, res2.pvalue)