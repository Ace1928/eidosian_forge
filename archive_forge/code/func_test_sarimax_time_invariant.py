from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
@pytest.mark.parametrize('revisions', [True, False])
@pytest.mark.parametrize('updates', [True, False])
@pytest.mark.parametrize('revisions_details_start', [True, False, -2])
def test_sarimax_time_invariant(revisions, updates, revisions_details_start):
    endog = dta['infl'].copy()
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
    mod = sarimax.SARIMAX(endog1)
    res = mod.smooth([0.5, 1.0])
    news = res.news(endog2, start='2009Q2', end='2010Q1', comparison_type=comparison_type, revisions_details_start=revisions_details_start)
    impact_dates = pd.period_range(start='2009Q2', end='2010Q1', freq='Q')
    impacted_variables = ['infl']
    if revisions and updates:
        revisions_index = pd.MultiIndex.from_arrays([endog1.index[-1:], ['infl']], names=['revision date', 'revised variable'])
        revision_impacts = endog2.iloc[-2] * 0.5 ** np.arange(4).reshape(4, 1)
    elif revisions:
        revisions_index = pd.MultiIndex.from_arrays([endog1.index[-1:], ['infl']], names=['revision date', 'revised variable'])
        revision_impacts = np.r_[0, endog2.iloc[-1] * 0.5 ** np.arange(3)].reshape(4, 1)
    else:
        revisions_index = pd.MultiIndex.from_arrays([[], []], names=['revision date', 'revised variable'])
        revision_impacts = None
    if updates:
        updates_index = pd.MultiIndex.from_arrays([pd.period_range(start='2009Q3', periods=1, freq='Q'), ['infl']], names=['update date', 'updated variable'])
        update_impacts = np.array([[0, endog.loc['2009Q3'] - 0.5 * endog.loc['2009Q2'], 0.5 * endog.loc['2009Q3'] - 0.5 ** 2 * endog.loc['2009Q2'], 0.5 ** 2 * endog.loc['2009Q3'] - 0.5 ** 3 * endog.loc['2009Q2']]]).T
    else:
        updates_index = pd.MultiIndex.from_arrays([[], []], names=['update date', 'updated variable'])
        update_impacts = None
    print(update_impacts)
    if updates:
        prev_impacted_forecasts = np.r_[endog1.iloc[-1] * 0.5 ** np.arange(4)].reshape(4, 1)
    else:
        prev_impacted_forecasts = np.r_[endog1.iloc[-2], endog1.iloc[-1] * 0.5 ** np.arange(3)].reshape(4, 1)
    post_impacted_forecasts = np.r_[endog2.iloc[-2], 0.5 ** np.arange(3) * endog2.iloc[-1]].reshape(4, 1)
    if updates:
        update_forecasts = [0.5 * endog2.loc['2009Q2']]
        update_realized = [endog2.loc['2009Q3']]
        news_desired = [update_realized[i] - update_forecasts[i] for i in range(len(update_forecasts))]
        weights = pd.DataFrame(np.r_[0, 0.5 ** np.arange(3)]).T
    else:
        update_forecasts = pd.Series([], dtype=np.float64)
        update_realized = pd.Series([], dtype=np.float64)
        news_desired = pd.Series([], dtype=np.float64)
        weights = pd.DataFrame(np.zeros((0, 4)))
    check_news(news, revisions, updates, impact_dates, impacted_variables, revisions_index, updates_index, revision_impacts, update_impacts, prev_impacted_forecasts, post_impacted_forecasts, update_forecasts, update_realized, news_desired, weights)