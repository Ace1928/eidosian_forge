import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.arrays.boolean import coerce_to_array
def test_to_boolean_array_integer_like():
    result = pd.array([1, 0, 1, 0], dtype='boolean')
    expected = pd.array([True, False, True, False], dtype='boolean')
    tm.assert_extension_array_equal(result, expected)
    result = pd.array([1, 0, 1, None], dtype='boolean')
    expected = pd.array([True, False, True, None], dtype='boolean')
    tm.assert_extension_array_equal(result, expected)