from pathlib import Path
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.tsa.seasonal import MSTL
def test_return_pandas_series_when_input_pandas_and_len_periods_one(data_pd):
    mod = MSTL(endog=data_pd, periods=5)
    res = mod.fit()
    assert isinstance(res.trend, pd.Series)
    assert isinstance(res.seasonal, pd.Series)
    assert isinstance(res.resid, pd.Series)
    assert isinstance(res.weights, pd.Series)