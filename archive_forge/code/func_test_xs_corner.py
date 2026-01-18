import re
import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_xs_corner(self):
    df = DataFrame(index=[0])
    df['A'] = 1.0
    df['B'] = 'foo'
    df['C'] = 2.0
    df['D'] = 'bar'
    df['E'] = 3.0
    xs = df.xs(0)
    exp = Series([1.0, 'foo', 2.0, 'bar', 3.0], index=list('ABCDE'), name=0)
    tm.assert_series_equal(xs, exp)
    df = DataFrame(index=['a', 'b', 'c'])
    result = df.xs('a')
    expected = Series([], name='a', dtype=np.float64)
    tm.assert_series_equal(result, expected)