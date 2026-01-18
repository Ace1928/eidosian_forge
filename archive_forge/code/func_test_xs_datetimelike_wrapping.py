import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_xs_datetimelike_wrapping():
    arr = date_range('2016-01-01', periods=3)._data._ndarray
    ser = Series(arr, dtype=object)
    for i in range(len(ser)):
        ser.iloc[i] = arr[i]
    assert ser.dtype == object
    assert isinstance(ser[0], np.datetime64)
    result = ser.xs(0)
    assert isinstance(result, np.datetime64)