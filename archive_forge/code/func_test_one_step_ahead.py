from statsmodels.compat.pandas import QUARTER_END
from statsmodels.compat.platform import PLATFORM_LINUX32, PLATFORM_WIN
from itertools import product
import json
import pathlib
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import statsmodels.tsa.holtwinters as holtwinters
import statsmodels.tsa.statespace.exponential_smoothing as statespace
def test_one_step_ahead(setup_model):
    model, params, results_R = setup_model
    model2 = ETSModel(pd.Series(model.endog), seasonal_periods=model.seasonal_periods, error=model.error, trend=model.trend, seasonal=model.seasonal, damped_trend=model.damped_trend)
    res = model2.smooth(params)
    fcast1 = res.forecast(steps=1)
    fcast2 = res.forecast(steps=2)
    assert_allclose(fcast1.iloc[0], fcast2.iloc[0])
    pred1 = res.get_prediction(start=model2.nobs, end=model2.nobs, simulate_repetitions=2)
    pred2 = res.get_prediction(start=model2.nobs, end=model2.nobs + 1, simulate_repetitions=2)
    df1 = pred1.summary_frame(alpha=0.05)
    df2 = pred1.summary_frame(alpha=0.05)
    assert_allclose(df1.iloc[0, 0], df2.iloc[0, 0])