from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests import test_dynamic_factor_mq_monte_carlo
@pytest.mark.parametrize('idiosyncratic_ar1', [True, False])
def test_standardized_MQ(reset_randomstate, idiosyncratic_ar1):
    nobs = 100
    idiosyncratic_ar1 = False
    k1 = 2
    k2 = 2
    endog1_M, endog1_Q, f1 = test_dynamic_factor_mq_monte_carlo.gen_k_factor1(nobs, k=k1, idiosyncratic_ar1=idiosyncratic_ar1)
    endog2_M, endog2_Q, f2 = test_dynamic_factor_mq_monte_carlo.gen_k_factor2(nobs, k=k2, idiosyncratic_ar1=idiosyncratic_ar1)
    endog_M = pd.concat([endog1_M, f2, endog2_M], axis=1, sort=True)
    endog_Q = pd.concat([endog1_Q, endog2_Q], axis=1, sort=True)
    endog_M1 = endog_M.loc[:'1957-12']
    endog_Q1 = endog_Q.loc[:'1957Q4']
    endog_M2 = endog_M.loc['1958-01':]
    endog_Q2 = endog_Q.loc['1958Q1':]
    factors = {f'yM{i + 1}_f1': ['a'] for i in range(k1)}
    factors.update({f'f{i + 1}': ['b'] for i in range(2)})
    factors.update({f'yM{i + 1}_f2': ['b'] for i in range(k2)})
    factors.update({f'yQ{i + 1}_f1': ['a'] for i in range(k1)})
    factors.update({f'yQ{i + 1}_f2': ['b'] for i in range(k2)})
    factor_multiplicities = {'b': 2}
    endog_mean = pd.Series(np.random.normal(size=len(factors)), index=factors.keys())
    endog_std = pd.Series(np.abs(np.random.normal(size=len(factors))), index=factors.keys())
    mod1 = dynamic_factor_mq.DynamicFactorMQ(endog_M1, endog_quarterly=endog_Q1, factors=factors, factor_multiplicities=factor_multiplicities, factor_orders=6, idiosyncratic_ar1=idiosyncratic_ar1, standardize=(endog_mean, endog_std))
    params = pd.Series(mod1.start_params, index=mod1.param_names)
    res1 = mod1.smooth(params)
    mod2 = dynamic_factor_mq.DynamicFactorMQ(endog_M1, endog_quarterly=endog_Q1, factors=factors, factor_multiplicities=factor_multiplicities, factor_orders=6, idiosyncratic_ar1=idiosyncratic_ar1, standardize=False)
    mod2.update(params)
    mod2['obs_intercept'] = endog_mean
    mod2['design'] *= np.array(endog_std)[:, None]
    mod2['obs_cov'] *= np.array(endog_std)[:, None] ** 2
    mod2.update = lambda params, **kwargs: params
    res2 = mod2.smooth(params)
    check_standardized_results(res1, res2)
    check_append(res1, res2, endog_M2, endog_Q2)
    check_extend(res1, res2, endog_M2, endog_Q2)
    check_apply(res1, res2, endog_M, endog_Q)
    res1_apply = res1.apply(endog_M, endog_quarterly=endog_Q)
    res2_apply = res2.apply(endog_M, endog_quarterly=endog_Q)
    mod2_apply = res2_apply.model
    mod2_apply.update(res2_apply.params)
    mod2_apply['obs_intercept'] = mod2['obs_intercept']
    mod2_apply['design'] = mod2['design']
    mod2_apply['obs_cov'] = mod2['obs_cov']
    mod2_apply.update = lambda params, **kwargs: params
    res2_apply = mod2_apply.smooth(res2_apply.params)
    news1 = res1_apply.news(res1, start='1958-01', end='1958-03', comparison_type='previous')
    news2 = res2_apply.news(res2, start='1958-01', end='1958-03', comparison_type='previous')
    attributes = ['total_impacts', 'update_impacts', 'revision_impacts', 'news', 'weights', 'update_forecasts', 'update_realized', 'prev_impacted_forecasts', 'post_impacted_forecasts', 'revisions_iloc', 'revisions_ix', 'updates_iloc', 'updates_ix']
    for attr in attributes:
        w = getattr(news1, attr)
        x = getattr(news2, attr)
        if isinstance(x, pd.Series):
            assert_series_equal(w, x)
        else:
            assert_frame_equal(w, x)