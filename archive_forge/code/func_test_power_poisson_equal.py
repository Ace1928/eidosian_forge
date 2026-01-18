import pytest
import warnings
import numpy as np
from numpy import arange
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
import statsmodels.stats.rates as smr
from statsmodels.stats.rates import (
def test_power_poisson_equal():
    nobs1, nobs2 = (6, 8)
    nobs_ratio = nobs2 / nobs1
    rate1, rate2 = (15, 10)
    pow_ = power_poisson_diff_2indep(rate1, rate2, nobs1, nobs_ratio=nobs_ratio, alpha=0.05, value=0, method_var='alt', alternative='larger', return_results=True)
    assert_allclose(pow_.power, 0.82566, atol=5e-05)
    pow_ = power_poisson_diff_2indep(0.6, 0.6, 97, 3 / 2, value=0.3, alpha=0.025, alternative='smaller', method_var='score', return_results=True)
    assert_allclose(pow_.power, 0.802596, atol=5e-05)
    pow_ = power_poisson_diff_2indep(0.6, 0.6, 128, 2 / 3, value=0.3, alpha=0.025, alternative='smaller', method_var='score', return_results=True)
    assert_allclose(pow_.power, 0.80194, atol=5e-05)