import numpy as np
import pytest
from pandas.compat import IS64
import pandas as pd
import pandas._testing as tm
def test_value_counts_empty():
    ser = pd.Series([], dtype='Float64')
    result = ser.value_counts()
    idx = pd.Index([], dtype='Float64')
    assert idx.dtype == 'Float64'
    expected = pd.Series([], index=idx, dtype='Int64', name='count')
    tm.assert_series_equal(result, expected)