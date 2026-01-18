import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_regular_min(self):
    df = DataFrame({'A': date_range('20130101', periods=5, freq='s'), 'B': [0.0, 1, 2, 3, 4]}).set_index('A')
    result = df.rolling('1s').min()
    expected = df.copy()
    expected['B'] = [0.0, 1, 2, 3, 4]
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'A': date_range('20130101', periods=5, freq='s'), 'B': [5, 4, 3, 4, 5]}).set_index('A')
    tm.assert_frame_equal(result, expected)
    result = df.rolling('2s').min()
    expected = df.copy()
    expected['B'] = [5.0, 4, 3, 3, 4]
    tm.assert_frame_equal(result, expected)
    result = df.rolling('5s').min()
    expected = df.copy()
    expected['B'] = [5.0, 4, 3, 3, 3]
    tm.assert_frame_equal(result, expected)