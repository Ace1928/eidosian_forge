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
def test_wtd_patsy_missing():
    import pandas as pd
    data = cpunish.load()
    data.endog = np.require(data.endog, requirements='W')
    data.exog = np.require(data.exog, requirements='W')
    data.exog[0, 0] = np.nan
    data.endog[[2, 4, 6, 8]] = np.nan
    data.pandas = pd.DataFrame(data.exog, columns=data.exog_name)
    data.pandas['EXECUTIONS'] = data.endog
    weights = np.arange(1, len(data.endog) + 1)
    formula = 'EXECUTIONS ~ INCOME + PERPOVERTY + PERBLACK + VC100k96 +\n                 SOUTH + DEGREE'
    mod_misisng = GLM.from_formula(formula, data=data.pandas, freq_weights=weights)
    assert_equal(mod_misisng.freq_weights.shape[0], mod_misisng.endog.shape[0])
    assert_equal(mod_misisng.freq_weights.shape[0], mod_misisng.exog.shape[0])
    assert_equal(mod_misisng.freq_weights.shape[0], 12)
    keep_weights = np.array([2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17])
    assert_equal(mod_misisng.freq_weights, keep_weights)