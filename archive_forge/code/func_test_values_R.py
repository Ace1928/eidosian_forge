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
def test_values_R(self):
    res = cramervonmises([-1.7, 2, 0, 1.3, 4, 0.1, 0.6], 'norm')
    assert_allclose(res.statistic, 0.288156, atol=1e-06)
    assert_allclose(res.pvalue, 0.1453465, atol=1e-06)
    res = cramervonmises([-1.7, 2, 0, 1.3, 4, 0.1, 0.6], 'norm', (3, 1.5))
    assert_allclose(res.statistic, 0.9426685, atol=1e-06)
    assert_allclose(res.pvalue, 0.002026417, atol=1e-06)
    res = cramervonmises([1, 2, 5, 1.4, 0.14, 11, 13, 0.9, 7.5], 'expon')
    assert_allclose(res.statistic, 0.8421854, atol=1e-06)
    assert_allclose(res.pvalue, 0.004433406, atol=1e-06)