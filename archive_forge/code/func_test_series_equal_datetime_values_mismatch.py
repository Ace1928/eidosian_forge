import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_series_equal_datetime_values_mismatch(rtol):
    msg = 'Series are different\n\nSeries values are different \\(100.0 %\\)\n\\[index\\]: \\[0, 1, 2\\]\n\\[left\\]:  \\[1514764800000000000, 1514851200000000000, 1514937600000000000\\]\n\\[right\\]: \\[1549065600000000000, 1549152000000000000, 1549238400000000000\\]'
    s1 = Series(pd.date_range('2018-01-01', periods=3, freq='D'))
    s2 = Series(pd.date_range('2019-02-02', periods=3, freq='D'))
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s2, rtol=rtol)