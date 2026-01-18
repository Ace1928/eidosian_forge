import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.datasets import cpunish, longley
from statsmodels.discrete import discrete_model as discrete
from statsmodels.genmod.generalized_linear_model import GLM, SET_USE_BIC_LLF
from statsmodels.tools.numdiff import (
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import add_constant
@pytest.mark.filterwarnings('ignore:GLM ridge optimization')
def test_tweedie_EQL():
    p = 1.5
    y, x = gen_tweedie(p)
    fam = sm.families.Tweedie(var_power=p, eql=True)
    model1 = sm.GLM(y, x, family=fam)
    result1 = model1.fit(method='newton')
    assert_allclose(result1.params, np.array([1.00350497, -0.99656954, 0.00802702, 0.50713209]), rtol=1e-05, atol=1e-05)
    model1x = sm.GLM(y, x, family=fam)
    result1x = model1x.fit(method='irls')
    assert_allclose(result1.params, result1x.params)
    assert_allclose(result1.bse, result1x.bse, rtol=0.01)
    model2 = sm.GLM(y, x, family=fam)
    result2 = model2.fit_regularized(L1_wt=1, alpha=0.07, maxiter=200, cnvrg_tol=0.01)
    rtol, atol = (0.01, 0.0001)
    assert_allclose(result2.params, np.array([0.976831, -0.952854, 0.0, 0.470171]), rtol=rtol, atol=atol)
    ev = (np.array([1.001778, -0.99388, 0.00797, 0.506183]), np.array([0.98586638, -0.96953481, 0.00749983, 0.4975267]), np.array([0.206429, -0.164547, 0.000235, 0.102489]))
    for j, alpha in enumerate([0.05, 0.5, 0.7]):
        model3 = sm.GLM(y, x, family=fam)
        result3 = model3.fit_regularized(L1_wt=0, alpha=alpha)
        assert_allclose(result3.params, ev[j], rtol=rtol, atol=atol)
        result4 = model3.fit_regularized(L1_wt=0, alpha=alpha * np.ones(x.shape[1]))
        assert_allclose(result4.params, result3.params, rtol=rtol, atol=atol)
        alpha = alpha * np.ones(x.shape[1])
        alpha[0] = 0
        result5 = model3.fit_regularized(L1_wt=0, alpha=alpha)
        assert not np.allclose(result5.params, result4.params)