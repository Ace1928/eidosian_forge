from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
@pytest.mark.parametrize('revisions', [True, False])
@pytest.mark.parametrize('updates', [True, False])
@pytest.mark.parametrize('which', ['exog', 'trend'])
def test_sarimax_time_varying(revisions, updates, which):
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
    exog1 = None
    exog2 = None
    trend = 'n'
    if which == 'exog':
        exog1 = np.ones_like(endog1)
        exog2 = np.ones_like(endog2)
    elif which == 'trend':
        trend = 't'
    mod1 = sarimax.SARIMAX(endog1, exog=exog1, trend=trend)
    res1 = mod1.smooth([0.0, 0.5, 1.0])
    news1 = res1.news(endog2, exog=exog2, start='2008Q1', end='2009Q3', comparison_type=comparison_type)
    mod2 = sarimax.SARIMAX(endog1)
    res2 = mod2.smooth([0.5, 1.0])
    news2 = res2.news(endog2, start='2008Q1', end='2009Q3', comparison_type=comparison_type)
    attrs = ['total_impacts', 'update_impacts', 'revision_impacts', 'news', 'weights', 'update_forecasts', 'update_realized', 'prev_impacted_forecasts', 'post_impacted_forecasts', 'revisions_iloc', 'revisions_ix', 'updates_iloc', 'updates_ix']
    for attr in attrs:
        w = getattr(news1, attr)
        x = getattr(news2, attr)
        if isinstance(x, pd.Series):
            assert_series_equal(w, x)
        else:
            assert_frame_equal(w, x)