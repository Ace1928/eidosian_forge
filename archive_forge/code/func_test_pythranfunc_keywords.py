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
def test_pythranfunc_keywords(self):
    table = [[27, 25, 14, 7, 0], [7, 14, 18, 35, 12], [1, 3, 2, 7, 17]]
    res1 = stats.somersd(table)
    optional_args = self.get_optional_args(stats.somersd)
    res2 = stats.somersd(table, **optional_args)
    assert_allclose(res1.statistic, res2.statistic, atol=1e-15)
    assert_allclose(res1.pvalue, res2.pvalue, atol=1e-15)