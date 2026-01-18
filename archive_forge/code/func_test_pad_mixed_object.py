from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
def test_pad_mixed_object():
    s = Series(['a', np.nan, 'b', True, datetime.today(), 'ee', None, 1, 2.0])
    result = s.str.pad(5, side='left')
    expected = Series(['    a', np.nan, '    b', np.nan, np.nan, '   ee', None, np.nan, np.nan], dtype=object)
    tm.assert_series_equal(result, expected)
    result = s.str.pad(5, side='right')
    expected = Series(['a    ', np.nan, 'b    ', np.nan, np.nan, 'ee   ', None, np.nan, np.nan], dtype=object)
    tm.assert_series_equal(result, expected)
    result = s.str.pad(5, side='both')
    expected = Series(['  a  ', np.nan, '  b  ', np.nan, np.nan, '  ee ', None, np.nan, np.nan], dtype=object)
    tm.assert_series_equal(result, expected)