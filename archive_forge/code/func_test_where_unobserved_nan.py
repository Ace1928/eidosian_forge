import math
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_where_unobserved_nan(self):
    ser = Series(Categorical(['a', 'b']))
    result = ser.where([True, False])
    expected = Series(Categorical(['a', None], categories=['a', 'b']))
    tm.assert_series_equal(result, expected)
    ser = Series(Categorical(['a', 'b']))
    result = ser.where([False, False])
    expected = Series(Categorical([None, None], categories=['a', 'b']))
    tm.assert_series_equal(result, expected)