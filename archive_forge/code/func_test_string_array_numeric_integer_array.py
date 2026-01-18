import numpy as np
import pytest
from pandas._libs import lib
from pandas import (
@pytest.mark.parametrize('method,expected', [('count', [2, None]), ('find', [0, None]), ('index', [0, None]), ('rindex', [2, None])])
def test_string_array_numeric_integer_array(nullable_string_dtype, method, expected):
    s = Series(['aba', None], dtype=nullable_string_dtype)
    result = getattr(s.str, method)('a')
    expected = Series(expected, dtype='Int64')
    tm.assert_series_equal(result, expected)