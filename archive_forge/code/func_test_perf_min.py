import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_perf_min(self):
    N = 10000
    dfp = DataFrame({'B': np.random.default_rng(2).standard_normal(N)}, index=date_range('20130101', periods=N, freq='s'))
    expected = dfp.rolling(2, min_periods=1).min()
    result = dfp.rolling('2s').min()
    assert (result - expected < 0.01).all().all()
    expected = dfp.rolling(200, min_periods=1).min()
    result = dfp.rolling('200s').min()
    assert (result - expected < 0.01).all().all()