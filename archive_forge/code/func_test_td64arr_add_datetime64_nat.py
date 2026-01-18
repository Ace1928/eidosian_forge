from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_add_datetime64_nat(self, box_with_array):
    other = np.datetime64('NaT')
    tdi = timedelta_range('1 day', periods=3)
    expected = DatetimeIndex(['NaT', 'NaT', 'NaT'], dtype='M8[ns]')
    tdser = tm.box_expected(tdi, box_with_array)
    expected = tm.box_expected(expected, box_with_array)
    tm.assert_equal(tdser + other, expected)
    tm.assert_equal(other + tdser, expected)