import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_rolling_on_decreasing_index(self, unit):
    index = DatetimeIndex([Timestamp('20190101 09:00:30'), Timestamp('20190101 09:00:27'), Timestamp('20190101 09:00:20'), Timestamp('20190101 09:00:18'), Timestamp('20190101 09:00:10')]).as_unit(unit)
    df = DataFrame({'column': [3, 4, 4, 5, 6]}, index=index)
    result = df.rolling('5s').min()
    expected = DataFrame({'column': [3.0, 3.0, 4.0, 4.0, 6.0]}, index=index)
    tm.assert_frame_equal(result, expected)