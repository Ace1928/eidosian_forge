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
@pytest.fixture(scope='module', params=params, ids=ids)
def ar_data(request):
    lags, trend, seasonal = request.param[:3]
    nexog, period, missing, use_pandas, hold_back = request.param[3:]
    data = gen_data(250, nexog, use_pandas)
    return Bunch(trend=trend, lags=lags, seasonal=seasonal, period=period, endog=data.endog, exog=data.exog, missing=missing, hold_back=hold_back)