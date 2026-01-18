import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('input_array,expected', [([1 + 1j, 1 + 1j, 1, 3j, 3j, 3j], Series([3, 2, 1], index=Index([3j, 1 + 1j, 1], dtype=np.complex128), name='count')), (np.array([1 + 1j, 1 + 1j, 1, 3j, 3j, 3j], dtype=np.complex64), Series([3, 2, 1], index=Index([3j, 1 + 1j, 1], dtype=np.complex64), name='count'))])
def test_value_counts_complex_numbers(self, input_array, expected):
    result = Series(input_array).value_counts()
    tm.assert_series_equal(result, expected)