import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_take_non_na_fill_value(self, data_missing):
    fill_value = data_missing[1]
    na = data_missing[0]
    arr = data_missing._from_sequence([na, fill_value, na], dtype=data_missing.dtype)
    result = arr.take([-1, 1], fill_value=fill_value, allow_fill=True)
    expected = arr.take([1, 1])
    tm.assert_extension_array_equal(result, expected)