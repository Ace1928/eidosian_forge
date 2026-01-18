from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
def test_pipe_failures(any_string_dtype):
    ser = Series(['A|B|C'], dtype=any_string_dtype)
    result = ser.str.split('|')
    expected = Series([['A', 'B', 'C']], dtype=object)
    tm.assert_series_equal(result, expected)
    result = ser.str.replace('|', ' ', regex=False)
    expected = Series(['A B C'], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)