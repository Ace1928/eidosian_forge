import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
def test_series_apply_unpack_nested_data():
    ser = Series([[1, 2, 3], [4, 5, 6, 7]])
    result = ser.apply(lambda x: Series(x))
    expected = DataFrame({0: [1.0, 4.0], 1: [2.0, 5.0], 2: [3.0, 6.0], 3: [np.nan, 7]})
    tm.assert_frame_equal(result, expected)