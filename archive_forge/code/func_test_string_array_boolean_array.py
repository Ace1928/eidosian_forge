import numpy as np
import pytest
from pandas._libs import lib
from pandas import (
@pytest.mark.parametrize('method,expected', [('isdigit', [False, None, True]), ('isalpha', [True, None, False]), ('isalnum', [True, None, True]), ('isnumeric', [False, None, True])])
def test_string_array_boolean_array(nullable_string_dtype, method, expected):
    s = Series(['a', None, '1'], dtype=nullable_string_dtype)
    result = getattr(s.str, method)()
    expected = Series(expected, dtype='boolean')
    tm.assert_series_equal(result, expected)