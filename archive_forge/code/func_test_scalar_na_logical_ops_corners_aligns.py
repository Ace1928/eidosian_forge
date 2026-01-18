from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_scalar_na_logical_ops_corners_aligns(self):
    s = Series([2, 3, 4, 5, 6, 7, 8, 9, datetime(2005, 1, 1)])
    s[::2] = np.nan
    d = DataFrame({'A': s})
    expected = DataFrame(False, index=range(9), columns=['A'] + list(range(9)))
    result = s & d
    tm.assert_frame_equal(result, expected)
    result = d & s
    tm.assert_frame_equal(result, expected)