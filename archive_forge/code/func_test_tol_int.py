import pytest
import warnings
import numpy as np
from numpy import arange
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
import statsmodels.stats.rates as smr
from statsmodels.stats.rates import (
@pytest.mark.parametrize('case', cases_tolint)
def test_tol_int(case):
    prob = 0.95
    prob_one = 0.975
    meth, count, exposure, exposure_new, r2, rs, rl = case
    ti = tolerance_int_poisson(count, exposure, prob, exposure_new=exposure_new, method=meth, alpha=0.05, alternative='two-sided')
    assert_equal(ti, r2)
    ti = tolerance_int_poisson(count, exposure, prob_one, exposure_new=exposure_new, method=meth, alpha=0.05, alternative='larger')
    assert_equal(ti, rl)
    ti = tolerance_int_poisson(count, exposure, prob_one, exposure_new=exposure_new, method=meth, alpha=0.05, alternative='smaller')
    assert_equal(ti, rs)
    if meth not in ['exact-c']:
        ti = tolerance_int_poisson(count, exposure, prob, exposure_new=exposure_new, method=meth, alpha=0.99999, alternative='two-sided')
        ci = stats.poisson.interval(prob, count / exposure * exposure_new)
        assert_equal(ti, ci)
    ciq = confint_quantile_poisson(count, exposure, prob_one, exposure_new=exposure_new, method=meth, alpha=0.05, alternative='two-sided')
    assert_equal(ciq[1], r2[1])
    ciq = confint_quantile_poisson(count, exposure, prob_one, exposure_new=exposure_new, method=meth, alpha=0.05, alternative='larger')
    assert_equal(ciq[1], rl[1])
    prob_low = 0.025
    ciq = confint_quantile_poisson(count, exposure, prob_low, exposure_new=exposure_new, method=meth, alpha=0.05, alternative='two-sided')
    assert_equal(ciq[0], r2[0])
    ciq = confint_quantile_poisson(count, exposure, prob_low, exposure_new=exposure_new, method=meth, alpha=0.05, alternative='smaller')
    assert_equal(ciq[0], rs[0])