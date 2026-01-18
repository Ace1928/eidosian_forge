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
def test_same_input(self):
    x = np.arange(15)
    res = cramervonmises_2samp(x, x)
    assert_equal((res.statistic, res.pvalue), (0.0, 1.0))
    res = cramervonmises_2samp(x[:4], x[:4])
    assert_equal((res.statistic, res.pvalue), (0.0, 1.0))