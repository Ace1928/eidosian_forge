from statsmodels.compat import lrange
import os
import numpy as np
import pytest
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
import statsmodels.genmod.generalized_estimating_equations as gee
import statsmodels.tools as tools
import statsmodels.regression.linear_model as lm
from statsmodels.genmod import families
from statsmodels.genmod import cov_struct
import statsmodels.discrete.discrete_model as discrete
import pandas as pd
from scipy.stats.distributions import norm
import warnings
def test_predict_exposure_lists(self):
    n = 50
    np.random.seed(34234)
    exog = [[1, np.random.normal(), np.random.normal()] for _ in range(n)]
    groups = list(np.kron(np.arange(25), np.r_[1, 1]))
    offset = list(np.random.uniform(1, 2, size=n))
    exposure = list(np.random.uniform(1, 2, size=n))
    endog = [np.random.poisson(0.1 * (exog_i[1] + exog_i[2]) + offset_i + np.log(exposure_i)) for exog_i, offset_i, exposure_i in zip(exog, offset, exposure)]
    model = gee.GEE(endog, exog, groups=groups, family=families.Poisson(), offset=offset, exposure=exposure)
    result = model.fit()
    pred1 = result.predict()
    pred2 = result.predict(exog=exog, offset=offset, exposure=exposure)
    assert_allclose(pred1, pred2)