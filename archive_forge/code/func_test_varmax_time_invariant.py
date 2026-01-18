from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
@pytest.mark.parametrize('revisions', [True, False])
@pytest.mark.parametrize('updates', [True, False])
def test_varmax_time_invariant(revisions, updates):
    endog = dta[['realgdp', 'unemp']].copy()
    endog['realgdp'] = np.log(endog['realgdp']).diff() * 400
    endog = endog.iloc[1:]
    comparison_type = None
    if updates:
        endog1 = endog.loc[:'2009Q2'].copy()
        endog2 = endog.loc[:'2009Q3'].copy()
    else:
        endog1 = endog.loc[:'2009Q3'].copy()
        endog2 = endog.loc[:'2009Q3'].copy()
        comparison_type = 'updated'
    if revisions:
        endog1.iloc[-1] = 0.0
    mod = varmax.VARMAX(endog1, trend='n')
    params = np.r_[0.5, 0.1, 0.2, 0.9, 1.0, 0.1, 1.1]
    res = mod.smooth(params)
    news = res.news(endog2, start='2009Q2', end='2010Q1', comparison_type=comparison_type)
    impact_dates = pd.period_range(start='2009Q2', end='2010Q1', freq='Q')
    impacted_variables = ['realgdp', 'unemp']
    Z = np.zeros((2, 2))
    T0 = np.eye(2)
    T1 = mod['transition']
    T2 = T1 @ T1
    T3 = T1 @ T2
    if revisions and updates:
        revisions_index = pd.MultiIndex.from_product([endog1.index[-1:], ['realgdp', 'unemp']], names=['revision date', 'revised variable'])
        tmp = endog2.iloc[-2].values
        revision_impacts = np.c_[T0 @ tmp, T1 @ tmp, T2 @ tmp, T3 @ tmp].T
    elif revisions:
        revisions_index = pd.MultiIndex.from_product([endog1.index[-1:], ['realgdp', 'unemp']], names=['revision date', 'revised variable'])
        tmp = endog2.iloc[-1].values
        revision_impacts = np.c_[Z @ tmp, T0 @ tmp, T1 @ tmp, T2 @ tmp].T
    else:
        revisions_index = pd.MultiIndex.from_product([[], []], names=['revision date', 'revised variable'])
        revision_impacts = None
    if updates:
        tmp = endog1.iloc[-1].values
        prev_impacted_forecasts = np.c_[T0 @ tmp, T1 @ tmp, T2 @ tmp, T3 @ tmp].T
        tmp = endog2.iloc[-2].values
        rev_impacted_forecasts = np.c_[T0 @ tmp, T1 @ tmp, T2 @ tmp, T3 @ tmp].T
    else:
        tmp = endog1.iloc[-1].values
        prev_impacted_forecasts = np.c_[T0 @ endog1.iloc[-2], T0 @ tmp, T1 @ tmp, T2 @ tmp].T
        tmp = endog2.iloc[-1].values
        rev_impacted_forecasts = np.c_[T0 @ endog2.iloc[-2], T0 @ tmp, T1 @ tmp, T2 @ tmp].T
    tmp = endog2.iloc[-1].values
    post_impacted_forecasts = np.c_[T0 @ endog2.iloc[-2], T0 @ tmp, T1 @ tmp, T2 @ tmp].T
    if updates:
        updates_index = pd.MultiIndex.from_product([pd.period_range(start='2009Q3', periods=1, freq='Q'), ['realgdp', 'unemp']], names=['update date', 'updated variable'])
        update_impacts = post_impacted_forecasts - rev_impacted_forecasts
    else:
        updates_index = pd.MultiIndex.from_product([[], []], names=['update date', 'updated variable'])
        update_impacts = None
    if updates:
        update_forecasts = T1 @ endog2.loc['2009Q2'].values
        update_realized = endog2.loc['2009Q3'].values
        news_desired = [update_realized[i] - update_forecasts[i] for i in range(len(update_forecasts))]
        columns = pd.MultiIndex.from_product([impact_dates, impacted_variables], names=['impact dates', 'impacted variables'])
        weights = pd.DataFrame(np.zeros((2, 8)), index=updates_index, columns=columns)
        weights.loc[:, '2009Q2'] = Z
        weights.loc[:, '2009Q3'] = T0
        weights.loc[:, '2009Q4'] = T1.T
        weights.loc[:, '2010Q1'] = T2.T
    else:
        update_forecasts = pd.Series([], dtype=np.float64)
        update_realized = pd.Series([], dtype=np.float64)
        news_desired = pd.Series([], dtype=np.float64)
        weights = pd.DataFrame(np.zeros((0, 8)))
    check_news(news, revisions, updates, impact_dates, impacted_variables, revisions_index, updates_index, revision_impacts, update_impacts, prev_impacted_forecasts, post_impacted_forecasts, update_forecasts, update_realized, news_desired, weights)