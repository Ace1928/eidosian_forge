from statsmodels.compat.pandas import deprecate_kwarg
from collections.abc import Iterable
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper
from statsmodels.stats._adnorm import anderson_statistic, normal_ad
from statsmodels.stats._lilliefors import (
from statsmodels.tools.validation import (
from statsmodels.tsa.tsatools import lagmat
def _check_nested_results(results_x, results_z):
    if not isinstance(results_x, RegressionResultsWrapper):
        raise TypeError('results_x must come from a linear regression model')
    if not isinstance(results_z, RegressionResultsWrapper):
        raise TypeError('results_z must come from a linear regression model')
    if not np.allclose(results_x.model.endog, results_z.model.endog):
        raise ValueError('endogenous variables in models are not the same')
    x = results_x.model.exog
    z = results_z.model.exog
    nested = False
    if x.shape[1] <= z.shape[1]:
        nested = nested or _check_nested_exog(x, z)
    else:
        nested = nested or _check_nested_exog(z, x)
    return nested