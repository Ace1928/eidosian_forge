import operator
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import TimedeltaArray
from pandas.tests.arithmetic.common import (
def test_parr_cmp_period_scalar2(self, box_with_array):
    pi = period_range('2000-01-01', periods=10, freq='D')
    val = pi[3]
    expected = [x > val for x in pi]
    ser = tm.box_expected(pi, box_with_array)
    xbox = get_upcast_box(ser, val, True)
    expected = tm.box_expected(expected, xbox)
    result = ser > val
    tm.assert_equal(result, expected)
    val = pi[5]
    result = ser > val
    expected = [x > val for x in pi]
    expected = tm.box_expected(expected, xbox)
    tm.assert_equal(result, expected)