import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_ragged_kurt(self, ragged):
    df = ragged
    result = df.rolling(window='3s', min_periods=1).kurt()
    expected = df.copy()
    expected['B'] = [np.nan] * 5
    tm.assert_frame_equal(result, expected)
    result = df.rolling(window='5s', min_periods=1).kurt()
    expected = df.copy()
    expected['B'] = [np.nan] * 4 + [-1.2]
    tm.assert_frame_equal(result, expected)