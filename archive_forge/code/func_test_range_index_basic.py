from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def test_range_index_basic():
    idx = pd.date_range('2000-1-1', freq=MONTH_END, periods=120)
    dp = DeterministicProcess(idx, constant=True, order=1, seasonal=True)
    dp.range('2001-1-1', '2008-1-1')
    dp.range('2001-1-1', '2015-1-1')
    dp.range('2013-1-1', '2008-1-1')
    dp.range(0, 100)
    dp.range(100, 150)
    dp.range(130, 150)
    with pytest.raises(ValueError):
        dp.range('1990-1-1', '2010-1-1')
    idx = pd.period_range('2000-1-1', freq='M', periods=120)
    dp = DeterministicProcess(idx, constant=True, order=1, seasonal=True)
    dp.range('2001-1-1', '2008-1-1')
    dp.range('2001-1-1', '2015-1-1')
    dp.range('2013-1-1', '2008-1-1')
    with pytest.raises(ValueError, match='start must be non-negative'):
        dp.range(-7, 200)
    dp.range(0, 100)
    dp.range(100, 150)
    dp.range(130, 150)
    idx = pd.RangeIndex(0, 120)
    dp = DeterministicProcess(idx, constant=True, order=1, seasonal=True, period=12)
    dp.range(0, 100)
    dp.range(100, 150)
    dp.range(120, 150)
    dp.range(130, 150)
    with pytest.raises(ValueError):
        dp.range(-10, 0)