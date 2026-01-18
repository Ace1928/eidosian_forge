from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_add_sub_tdi(self, box_with_array, names):
    box = box_with_array
    exname = get_expected_name(box, names)
    tdi = TimedeltaIndex(['0 days', '1 day'], name=names[1])
    tdi = np.array(tdi) if box in [tm.to_array, pd.array] else tdi
    ser = Series([Timedelta(hours=3), Timedelta(hours=4)], name=names[0])
    expected = Series([Timedelta(hours=3), Timedelta(days=1, hours=4)], name=exname)
    ser = tm.box_expected(ser, box)
    expected = tm.box_expected(expected, box)
    result = tdi + ser
    tm.assert_equal(result, expected)
    assert_dtype(result, 'timedelta64[ns]')
    result = ser + tdi
    tm.assert_equal(result, expected)
    assert_dtype(result, 'timedelta64[ns]')
    expected = Series([Timedelta(hours=-3), Timedelta(days=1, hours=-4)], name=exname)
    expected = tm.box_expected(expected, box)
    result = tdi - ser
    tm.assert_equal(result, expected)
    assert_dtype(result, 'timedelta64[ns]')
    result = ser - tdi
    tm.assert_equal(result, -expected)
    assert_dtype(result, 'timedelta64[ns]')