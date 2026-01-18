import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
from pandas.core.arrays.floating import (
def test_to_array_mixed_integer_float():
    result = pd.array([1, 2.0])
    expected = pd.array([1.0, 2.0], dtype='Float64')
    tm.assert_extension_array_equal(result, expected)
    result = pd.array([1, None, 2.0])
    expected = pd.array([1.0, None, 2.0], dtype='Float64')
    tm.assert_extension_array_equal(result, expected)