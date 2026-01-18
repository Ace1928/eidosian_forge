from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
def test_swapcase_mixed_object():
    s = Series(['FOO', np.nan, 'bar', True, datetime.today(), 'Blah', None, 1, 2.0])
    result = s.str.swapcase()
    expected = Series(['foo', np.nan, 'BAR', np.nan, np.nan, 'bLAH', None, np.nan, np.nan], dtype=object)
    tm.assert_series_equal(result, expected)