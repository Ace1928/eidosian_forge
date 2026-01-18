from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('left,right', [('lt', 'gt'), ('le', 'ge'), ('eq', 'eq'), ('ne', 'ne')])
def test_timestamp_compare_series(self, left, right):
    ser = Series(date_range('20010101', periods=10), name='dates')
    s_nat = ser.copy(deep=True)
    ser[0] = Timestamp('nat')
    ser[3] = Timestamp('nat')
    left_f = getattr(operator, left)
    right_f = getattr(operator, right)
    expected = left_f(ser, Timestamp('20010109'))
    result = right_f(Timestamp('20010109'), ser)
    tm.assert_series_equal(result, expected)
    expected = left_f(ser, Timestamp('nat'))
    result = right_f(Timestamp('nat'), ser)
    tm.assert_series_equal(result, expected)
    expected = left_f(s_nat, Timestamp('20010109'))
    result = right_f(Timestamp('20010109'), s_nat)
    tm.assert_series_equal(result, expected)
    expected = left_f(s_nat, NaT)
    result = right_f(NaT, s_nat)
    tm.assert_series_equal(result, expected)