from statsmodels.compat.pandas import MONTH_END, QUARTER_END, YEAR_END
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.tsa.seasonal import seasonal_decompose
def test_pandas_nofreq(self, reset_randomstate):
    nobs = 100
    dta = pd.Series([x % 3 for x in range(nobs)] + np.random.randn(nobs))
    res_np = seasonal_decompose(dta.values, period=3)
    res = seasonal_decompose(dta, period=3)
    atol = 1e-08
    rtol = 1e-10
    assert_allclose(res.seasonal.values.squeeze(), res_np.seasonal, atol=atol, rtol=rtol)
    assert_allclose(res.trend.values.squeeze(), res_np.trend, atol=atol, rtol=rtol)
    assert_allclose(res.resid.values.squeeze(), res_np.resid, atol=atol, rtol=rtol)