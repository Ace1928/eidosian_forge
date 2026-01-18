import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_series_equal_length_mismatch(rtol):
    msg = 'Series are different\n\nSeries length are different\n\\[left\\]:  3, RangeIndex\\(start=0, stop=3, step=1\\)\n\\[right\\]: 4, RangeIndex\\(start=0, stop=4, step=1\\)'
    s1 = Series([1, 2, 3])
    s2 = Series([1, 2, 3, 4])
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s2, rtol=rtol)