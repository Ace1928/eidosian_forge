import math
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_where_other_categorical(self):
    ser = Series(Categorical(['a', 'b', 'c'], categories=['d', 'c', 'b', 'a']))
    other = Categorical(['b', 'c', 'a'], categories=['a', 'c', 'b', 'd'])
    result = ser.where([True, False, True], other)
    expected = Series(Categorical(['a', 'c', 'c'], dtype=ser.dtype))
    tm.assert_series_equal(result, expected)