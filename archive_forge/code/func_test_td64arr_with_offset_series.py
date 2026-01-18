from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_with_offset_series(self, names, box_with_array):
    box = box_with_array
    box2 = Series if box in [Index, tm.to_array, pd.array] else box
    exname = get_expected_name(box, names)
    tdi = TimedeltaIndex(['1 days 00:00:00', '3 days 04:00:00'], name=names[0])
    other = Series([offsets.Hour(n=1), offsets.Minute(n=-2)], name=names[1])
    expected_add = Series([tdi[n] + other[n] for n in range(len(tdi))], name=exname, dtype=object)
    obj = tm.box_expected(tdi, box)
    expected_add = tm.box_expected(expected_add, box2).astype(object)
    with tm.assert_produces_warning(PerformanceWarning):
        res = obj + other
    tm.assert_equal(res, expected_add)
    with tm.assert_produces_warning(PerformanceWarning):
        res2 = other + obj
    tm.assert_equal(res2, expected_add)
    expected_sub = Series([tdi[n] - other[n] for n in range(len(tdi))], name=exname, dtype=object)
    expected_sub = tm.box_expected(expected_sub, box2).astype(object)
    with tm.assert_produces_warning(PerformanceWarning):
        res3 = obj - other
    tm.assert_equal(res3, expected_sub)