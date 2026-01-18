import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
def test_series_apply_map_box_timestamps(by_row):
    ser = Series(date_range('1/1/2000', periods=10))

    def func(x):
        return (x.hour, x.day, x.month)
    if not by_row:
        msg = "Series' object has no attribute 'hour'"
        with pytest.raises(AttributeError, match=msg):
            ser.apply(func, by_row=by_row)
        return
    result = ser.apply(func, by_row=by_row)
    expected = ser.map(func)
    tm.assert_series_equal(result, expected)