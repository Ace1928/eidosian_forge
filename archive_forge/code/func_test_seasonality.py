from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def test_seasonality(index):
    s = Seasonality(period=12)
    exog = s.in_sample(index)
    assert s.is_dummy
    assert exog.shape == (index.shape[0], 12)
    pd.testing.assert_index_equal(exog.index, index)
    assert np.all(exog.sum(1) == 1.0)
    assert list(exog.columns) == [f's({i},12)' for i in range(1, 13)]
    expected = np.zeros((index.shape[0], 12))
    for i in range(12):
        expected[i::12, i] = 1.0
    np.testing.assert_equal(expected, np.asarray(exog))
    warn = None
    if is_int_index(index) and np.any(np.diff(index) != 1) or (type(index) is pd.Index and max(index) > 2 ** 63):
        warn = UserWarning
    with pytest_warns(warn):
        fcast = s.out_of_sample(steps=12, index=index)
    assert fcast.iloc[0, len(index) % 12] == 1.0
    assert np.all(fcast.sum(1) == 1)
    s = Seasonality(period=7, initial_period=3)
    exog = s.in_sample(index)
    assert exog.iloc[0, 2] == 1.0
    assert exog.iloc[0].sum() == 1.0
    assert s.initial_period == 3
    with pytest.raises(ValueError, match='initial_period must be in'):
        Seasonality(period=12, initial_period=-3)
    with pytest.raises(ValueError, match='period must be >= 2'):
        Seasonality(period=1)