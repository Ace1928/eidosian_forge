import numpy as np
from pandas import (
import pandas._testing as tm
def test_unique_none(self):
    ser = Series([1, 2, 3, None, None, None], dtype=object)
    result = ser.unique()
    expected = np.array([1, 2, 3, None], dtype=object)
    tm.assert_numpy_array_equal(result, expected)