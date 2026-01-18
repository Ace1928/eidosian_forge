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
def test_rand_symm(self):
    np.random.seed(1234)
    data = np.random.rand(3, 100)
    res = stats.tukey_hsd(*data)
    conf = res.confidence_interval()
    assert_equal(conf.low, -conf.high.T)
    assert_equal(np.diagonal(conf.high), conf.high[0, 0])
    assert_equal(np.diagonal(conf.low), conf.low[0, 0])
    assert_equal(res.statistic, -res.statistic.T)
    assert_equal(np.diagonal(res.statistic), 0)
    assert_equal(res.pvalue, res.pvalue.T)
    assert_equal(np.diagonal(res.pvalue), 1)