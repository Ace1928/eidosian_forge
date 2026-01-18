import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.power as smpwr
import statsmodels.stats.oneway as smo  # needed for function with `test`
from statsmodels.stats.oneway import (
from statsmodels.stats.robust_compare import scale_transform
from statsmodels.stats.contrast import (
def test_oneway_effectsize():
    F = 5
    df1 = 3
    df2 = 76
    nobs = 80
    ci = confint_noncentrality(F, (df1, df2), alpha=0.05, alternative='two-sided')
    ci_es = confint_effectsize_oneway(F, (df1, df2), alpha=0.05)
    ci_steiger = ci_es.ci_f * np.sqrt(4 / 3)
    res_ci_steiger = [0.1764, 0.7367]
    res_ci_nc = np.asarray([1.8666, 32.563])
    assert_allclose(ci, res_ci_nc, atol=0.0001)
    assert_allclose(ci_es.ci_f_corrected, res_ci_steiger, atol=6e-05)
    assert_allclose(ci_steiger, res_ci_steiger, atol=6e-05)
    assert_allclose(ci_es.ci_f ** 2, res_ci_nc / nobs, atol=6e-05)
    assert_allclose(ci_es.ci_nc, res_ci_nc, atol=0.0001)