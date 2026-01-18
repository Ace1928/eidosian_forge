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
def test_output_exposure_null(reset_randomstate):
    x0 = [np.sin(i / 20) + 2 for i in range(1000)]
    rs = np.random.RandomState(0)
    exposure = rs.randint(100, 200, size=1000)
    y = [np.sum(rs.poisson(x, size=e)) for x, e in zip(x0, exposure)]
    x = add_constant(x0)
    model = GLM(endog=y, exog=x, exposure=exposure, family=sm.families.Poisson()).fit()
    null_model = GLM(endog=y, exog=x[:, 0], exposure=exposure, family=sm.families.Poisson()).fit()
    null_model_without_exposure = GLM(endog=y, exog=x[:, 0], family=sm.families.Poisson()).fit()
    assert_allclose(model.llnull, null_model.llf)
    assert np.abs(null_model_without_exposure.llf - model.llnull) > 1