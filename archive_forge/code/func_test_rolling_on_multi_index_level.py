import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_rolling_on_multi_index_level(self):
    df = DataFrame({'column': range(6)}, index=MultiIndex.from_product([date_range('20190101', periods=3), range(2)], names=['date', 'seq']))
    result = df.rolling('10d', on=df.index.get_level_values('date')).sum()
    expected = DataFrame({'column': [0.0, 1.0, 3.0, 6.0, 10.0, 15.0]}, index=df.index)
    tm.assert_frame_equal(result, expected)