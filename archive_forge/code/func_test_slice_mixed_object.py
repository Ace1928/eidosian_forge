from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
@pytest.mark.parametrize('start, stop, step, expected', [(2, 5, None, ['foo', np.nan, 'bar', np.nan, np.nan, None, np.nan, np.nan]), (4, 1, -1, ['oof', np.nan, 'rab', np.nan, np.nan, None, np.nan, np.nan])])
def test_slice_mixed_object(start, stop, step, expected):
    ser = Series(['aafootwo', np.nan, 'aabartwo', True, datetime.today(), None, 1, 2.0])
    result = ser.str.slice(start, stop, step)
    expected = Series(expected, dtype=object)
    tm.assert_series_equal(result, expected)