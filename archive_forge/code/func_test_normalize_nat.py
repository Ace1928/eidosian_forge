from dateutil.tz import tzlocal
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_normalize_nat(self):
    dti = DatetimeIndex([NaT, Timestamp('2018-01-01 01:00:00')])
    result = dti.normalize()
    expected = DatetimeIndex([NaT, Timestamp('2018-01-01')])
    tm.assert_index_equal(result, expected)