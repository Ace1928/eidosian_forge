from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
def test_wrap_unicode(any_string_dtype):
    s = Series(['  pre  ', np.nan, '¬€耀 abadcafe'], dtype=any_string_dtype)
    expected = Series(['  pre', np.nan, '¬€耀 ab\nadcafe'], dtype=any_string_dtype)
    result = s.str.wrap(6)
    tm.assert_series_equal(result, expected)