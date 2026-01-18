import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_shift_periods_freq():
    data = {'a': [1, 2, 3, 4, 5, 6], 'b': [0, 0, 0, 1, 1, 1]}
    df = DataFrame(data, index=date_range(start='20100101', periods=6))
    result = df.groupby(df.index).shift(periods=-2, freq='D')
    expected = DataFrame(data, index=date_range(start='2009-12-30', periods=6))
    tm.assert_frame_equal(result, expected)