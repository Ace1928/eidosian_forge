import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_size_groupby_all_null():
    df = DataFrame({'A': [None, None]})
    result = df.groupby('A').size()
    expected = Series(dtype='int64', index=Index([], name='A'))
    tm.assert_series_equal(result, expected)