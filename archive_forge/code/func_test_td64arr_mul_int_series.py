from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_mul_int_series(self, box_with_array, names):
    box = box_with_array
    exname = get_expected_name(box, names)
    tdi = TimedeltaIndex(['0days', '1day', '2days', '3days', '4days'], name=names[0])
    ser = Series([0, 1, 2, 3, 4], dtype=np.int64, name=names[1])
    expected = Series(['0days', '1day', '4days', '9days', '16days'], dtype='timedelta64[ns]', name=exname)
    tdi = tm.box_expected(tdi, box)
    xbox = get_upcast_box(tdi, ser)
    expected = tm.box_expected(expected, xbox)
    result = ser * tdi
    tm.assert_equal(result, expected)
    result = tdi * ser
    tm.assert_equal(result, expected)