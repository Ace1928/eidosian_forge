import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
@pytest.mark.parametrize('method', ['round', 'floor', 'ceil'])
@pytest.mark.parametrize('freq', ['s', '5s', 'min', '5min', 'h', '5h'])
def test_round_nat(self, method, freq, unit):
    ser = Series([pd.NaT], dtype=f'M8[{unit}]')
    expected = Series(pd.NaT, dtype=f'M8[{unit}]')
    round_method = getattr(ser.dt, method)
    result = round_method(freq)
    tm.assert_series_equal(result, expected)