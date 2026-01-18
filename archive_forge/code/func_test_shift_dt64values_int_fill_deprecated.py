import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_dt64values_int_fill_deprecated(self):
    ser = Series([pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02')])
    with pytest.raises(TypeError, match='value should be a'):
        ser.shift(1, fill_value=0)
    df = ser.to_frame()
    with pytest.raises(TypeError, match='value should be a'):
        df.shift(1, fill_value=0)
    df2 = DataFrame({'A': ser, 'B': ser})
    df2._consolidate_inplace()
    result = df2.shift(1, axis=1, fill_value=0)
    expected = DataFrame({'A': [0, 0], 'B': df2['A']})
    tm.assert_frame_equal(result, expected)
    df3 = DataFrame({'A': ser})
    df3['B'] = ser
    assert len(df3._mgr.arrays) == 2
    result = df3.shift(1, axis=1, fill_value=0)
    tm.assert_frame_equal(result, expected)