from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import datetime as dt
from itertools import product
from typing import NamedTuple, Union
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
from pandas import Index, Series, date_range, period_range
from pandas.testing import assert_series_equal
import pytest
from statsmodels.datasets import macrodata, sunspots
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.sm_exceptions import SpecificationWarning, ValueWarning
from statsmodels.tools.tools import Bunch
from statsmodels.tsa.ar_model import (
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.deterministic import (
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.tests.results import results_ar
def test_predict_errors():
    data = gen_data(250, 2, True)
    mod = AutoReg(data.endog, 3)
    res = mod.fit()
    with pytest.raises(ValueError, match='exog and exog_oos cannot be used'):
        mod.predict(res.params, exog=data.exog)
    with pytest.raises(ValueError, match='exog and exog_oos cannot be used'):
        mod.predict(res.params, exog_oos=data.exog)
    with pytest.raises(ValueError, match='hold_back must be >= lags'):
        AutoReg(data.endog, 3, hold_back=1)
    with pytest.raises(ValueError, match='freq cannot be inferred'):
        AutoReg(data.endog.values, 3, seasonal=True)
    mod = AutoReg(data.endog, 3, exog=data.exog)
    res = mod.fit()
    with pytest.raises(ValueError, match='The shape of exog \\(200, 2\\)'):
        mod.predict(res.params, exog=data.exog.iloc[:200])
    with pytest.raises(ValueError, match='The number of columns in exog_oos'):
        mod.predict(res.params, exog_oos=data.exog.iloc[:, :1])
    with pytest.raises(ValueError, match='Prediction must have `end` after'):
        mod.predict(res.params, start=200, end=199)
    with pytest.raises(ValueError, match='exog_oos must be provided'):
        mod.predict(res.params, end=250, exog_oos=None)
    mod = AutoReg(data.endog, 0, exog=data.exog)
    res = mod.fit()
    with pytest.raises(ValueError, match='start and end indicate that 10'):
        mod.predict(res.params, end=259, exog_oos=data.exog.iloc[:5])