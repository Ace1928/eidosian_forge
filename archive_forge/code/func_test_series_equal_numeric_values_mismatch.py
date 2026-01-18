import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_series_equal_numeric_values_mismatch(rtol):
    msg = 'Series are different\n\nSeries values are different \\(33\\.33333 %\\)\n\\[index\\]: \\[0, 1, 2\\]\n\\[left\\]:  \\[1, 2, 3\\]\n\\[right\\]: \\[1, 2, 4\\]'
    s1 = Series([1, 2, 3])
    s2 = Series([1, 2, 4])
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s2, rtol=rtol)