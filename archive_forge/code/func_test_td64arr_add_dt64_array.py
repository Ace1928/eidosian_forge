from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_add_dt64_array(self, box_with_array):
    dti = pd.date_range('2016-01-01', periods=3)
    tdi = TimedeltaIndex(['-1 Day'] * 3)
    dtarr = dti.values
    expected = DatetimeIndex(dtarr) + tdi
    tdi = tm.box_expected(tdi, box_with_array)
    expected = tm.box_expected(expected, box_with_array)
    result = tdi + dtarr
    tm.assert_equal(result, expected)
    result = dtarr + tdi
    tm.assert_equal(result, expected)