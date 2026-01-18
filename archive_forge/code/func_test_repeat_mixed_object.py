from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
def test_repeat_mixed_object():
    ser = Series(['a', np.nan, 'b', True, datetime.today(), 'foo', None, 1, 2.0])
    result = ser.str.repeat(3)
    expected = Series(['aaa', np.nan, 'bbb', np.nan, np.nan, 'foofoofoo', None, np.nan, np.nan], dtype=object)
    tm.assert_series_equal(result, expected)