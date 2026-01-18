import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_ragged_median(self, ragged):
    df = ragged
    result = df.rolling(window='1s', min_periods=1).median()
    expected = df.copy()
    expected['B'] = [0.0, 1, 2, 3, 4]
    tm.assert_frame_equal(result, expected)
    result = df.rolling(window='2s', min_periods=1).median()
    expected = df.copy()
    expected['B'] = [0.0, 1, 1.5, 3.0, 3.5]
    tm.assert_frame_equal(result, expected)