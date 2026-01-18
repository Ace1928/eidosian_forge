from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
def test_title_mixed_object():
    s = Series(['FOO', np.nan, 'bar', True, datetime.today(), 'blah', None, 1, 2.0])
    result = s.str.title()
    expected = Series(['Foo', np.nan, 'Bar', np.nan, np.nan, 'Blah', None, np.nan, np.nan], dtype=object)
    tm.assert_almost_equal(result, expected)