from pathlib import Path
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.tsa.seasonal import MSTL
@pytest.mark.parametrize('data, periods, windows, expected', [(data, 3, None, 1), (data, (3, 6), None, 2), (data, (3, 6, 1000000.0), None, 2)], indirect=['data'])
def test_number_of_seasonal_components(data, periods, windows, expected):
    mod = MSTL(endog=data, periods=periods, windows=windows)
    res = mod.fit()
    n_seasonal_components = res.seasonal.shape[1] if res.seasonal.ndim > 1 else res.seasonal.ndim
    assert n_seasonal_components == expected