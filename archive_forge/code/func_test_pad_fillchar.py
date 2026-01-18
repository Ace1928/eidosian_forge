from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
def test_pad_fillchar(any_string_dtype):
    s = Series(['a', 'b', np.nan, 'c', np.nan, 'eeeeee'], dtype=any_string_dtype)
    result = s.str.pad(5, side='left', fillchar='X')
    expected = Series(['XXXXa', 'XXXXb', np.nan, 'XXXXc', np.nan, 'eeeeee'], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)
    result = s.str.pad(5, side='right', fillchar='X')
    expected = Series(['aXXXX', 'bXXXX', np.nan, 'cXXXX', np.nan, 'eeeeee'], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)
    result = s.str.pad(5, side='both', fillchar='X')
    expected = Series(['XXaXX', 'XXbXX', np.nan, 'XXcXX', np.nan, 'eeeeee'], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)