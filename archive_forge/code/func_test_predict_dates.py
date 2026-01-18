import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_raises, assert_allclose, assert_
from statsmodels import datasets
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
def test_predict_dates():
    index = pd.date_range(start='1950-01-01', periods=11, freq='D')
    np.random.seed(324328)
    endog = pd.Series(np.random.normal(size=10), index=index[:-1])
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))
    res = mod.filter(mod.start_params)
    pred = res.predict()
    assert_equal(len(pred), mod.nobs)
    assert_equal(pred.index.values, index[:-1].values)
    fcast = res.forecast()
    assert_equal(fcast.index[0], index[-1])
    mod = sarimax.SARIMAX(endog, order=(1, 1, 0), simple_differencing=True)
    res = mod.filter(mod.start_params)
    pred = res.predict()
    assert_equal(mod.nobs, endog.shape[0] - 1)
    assert_equal(len(pred), mod.nobs)
    assert_equal(pred.index.values, index[1:-1].values)
    fcast = res.forecast()
    assert_equal(fcast.index[0], index[-1])
    mod = sarimax.SARIMAX(endog, order=(1, 2, 0), seasonal_order=(0, 1, 0, 4), simple_differencing=True)
    res = mod.filter(mod.start_params)
    pred = res.predict()
    assert_equal(mod.nobs, endog.shape[0] - (4 + 2))
    assert_equal(len(pred), mod.nobs)
    assert_equal(pred.index.values, index[4 + 2:-1].values)
    fcast = res.forecast()
    assert_equal(fcast.index[0], index[-1])