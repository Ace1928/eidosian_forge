import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_reindex_non_na_fill_value(self, data_missing):
    valid = data_missing[1]
    na = data_missing[0]
    arr = data_missing._from_sequence([na, valid], dtype=data_missing.dtype)
    ser = pd.Series(arr)
    result = ser.reindex([0, 1, 2], fill_value=valid)
    expected = pd.Series(data_missing._from_sequence([na, valid, valid], dtype=data_missing.dtype))
    tm.assert_series_equal(result, expected)