import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_series_equal_exact_for_nonnumeric():
    s1 = Series(['a', 'b'])
    s2 = Series(['a', 'b'])
    s3 = Series(['b', 'a'])
    tm.assert_series_equal(s1, s2, check_exact=True)
    tm.assert_series_equal(s2, s1, check_exact=True)
    msg = 'Series are different\n\nSeries values are different \\(100\\.0 %\\)\n\\[index\\]: \\[0, 1\\]\n\\[left\\]:  \\[a, b\\]\n\\[right\\]: \\[b, a\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s3, check_exact=True)
    msg = 'Series are different\n\nSeries values are different \\(100\\.0 %\\)\n\\[index\\]: \\[0, 1\\]\n\\[left\\]:  \\[b, a\\]\n\\[right\\]: \\[a, b\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s3, s1, check_exact=True)