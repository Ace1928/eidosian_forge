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
def test_tweedie_EQL_poisson_limit():
    np.random.seed(3242)
    n = 500
    x = np.random.normal(size=(n, 3))
    x[:, 0] = 1
    lpr = 4 + x[:, 1:].sum(1)
    mn = np.exp(lpr)
    y = np.random.poisson(mn)
    for scale in (1.0, 'x2', 'dev'):
        fam = sm.families.Tweedie(var_power=1, eql=True)
        model1 = sm.GLM(y, x, family=fam)
        result1 = model1.fit(method='newton', scale=scale)
        model2 = sm.GLM(y, x, family=sm.families.Poisson())
        result2 = model2.fit(method='newton', scale=scale)
        assert_allclose(result1.params, result2.params, atol=1e-06, rtol=1e-06)
        assert_allclose(result1.bse, result2.bse, 1e-06, 1e-06)