from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests import test_dynamic_factor_mq_monte_carlo
@pytest.mark.parametrize('use_pandas', [True, False])
@pytest.mark.parametrize('k_endog', [1, 2])
@pytest.mark.parametrize('idiosyncratic_ar1', [True, False])
def test_standardized_monthly(reset_randomstate, idiosyncratic_ar1, k_endog, use_pandas):
    nobs = 100
    k2 = 2
    _, _, f2 = test_dynamic_factor_mq_monte_carlo.gen_k_factor2(nobs, k=k2, idiosyncratic_ar1=idiosyncratic_ar1)
    if k_endog == 1:
        endog = f2.iloc[:, 0]
        endog_mean = pd.Series([10], index=['f1'])
        endog_std = pd.Series([1], index=['f1'])
    else:
        endog = f2
        endog_mean = pd.Series([10, -4], index=['f1', 'f2'])
        endog_std = pd.Series([1, 1], index=['f1', 'f2'])
    if not use_pandas:
        endog = endog.values
        endog_mean = endog_mean.values
        endog_std = endog_std.values
    mod1 = dynamic_factor_mq.DynamicFactorMQ(endog, factors=1, factor_multiplicities=1, factor_orders=1, idiosyncratic_ar1=idiosyncratic_ar1, standardize=(endog_mean, endog_std))
    params = pd.Series(mod1.start_params, index=mod1.param_names)
    res1 = mod1.smooth(params)
    mod2 = dynamic_factor_mq.DynamicFactorMQ(endog, factors=1, factor_multiplicities=1, factor_orders=1, idiosyncratic_ar1=idiosyncratic_ar1, standardize=False)
    mod2.update(params)
    mod2['obs_intercept'] = np.array(endog_mean)
    mod2['design'] *= np.array(endog_std)[:, None]
    mod2['obs_cov'] *= np.array(endog_std)[:, None] ** 2
    mod2.update = lambda params, **kwargs: params
    res2 = mod2.smooth(params)
    check_standardized_results(res1, res2)