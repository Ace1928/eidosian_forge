from statsmodels.compat.platform import PLATFORM_WIN32
import io
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose, assert_raises, assert_
from statsmodels.datasets import macrodata
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.estimators.yule_walker import yule_walker
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
from statsmodels.tsa.arima.estimators.statespace import statespace
def test_statespace():
    endog = dta['infl'].iloc[:100]
    desired_p, _ = statespace(endog, order=(1, 0, 1), include_constant=False)
    mod = ARIMA(endog, order=(1, 0, 1), trend='n')
    res = mod.fit(method='statespace')
    rtol = 1e-07 if not PLATFORM_WIN32 else 0.001
    assert_allclose(res.params, desired_p.params, rtol=rtol, atol=0.0001)
    desired_p, _ = statespace(endog, order=(1, 0, 2), include_constant=True)
    mod = ARIMA(endog, order=(1, 0, 2), trend='c')
    res = mod.fit(method='statespace')
    assert_allclose(res.params, desired_p.params, atol=0.0001)
    desired_p, _spec = statespace(endog, order=(1, 0, 0), seasonal_order=(1, 0, 0, 4), include_constant=False)
    mod = ARIMA(endog, order=(1, 0, 0), seasonal_order=(1, 0, 0, 4), trend='n')
    res = mod.fit(method='statespace')
    assert_allclose(res.params, desired_p.params, atol=0.0001)