import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
def test_take_categorical():
    ser = Series(pd.Categorical(['a', 'b', 'c']))
    result = ser.take([-2, -2, 0])
    expected = Series(pd.Categorical(['b', 'b', 'a'], categories=['a', 'b', 'c']), index=[1, 1, 0])
    tm.assert_series_equal(result, expected)