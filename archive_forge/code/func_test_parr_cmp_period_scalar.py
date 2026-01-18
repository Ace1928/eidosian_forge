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
@pytest.mark.parametrize('freq', ['M', '2M', '3M'])
def test_parr_cmp_period_scalar(self, freq, box_with_array):
    base = PeriodIndex(['2011-01', '2011-02', '2011-03', '2011-04'], freq=freq)
    base = tm.box_expected(base, box_with_array)
    per = Period('2011-02', freq=freq)
    xbox = get_upcast_box(base, per, True)
    exp = np.array([False, True, False, False])
    exp = tm.box_expected(exp, xbox)
    tm.assert_equal(base == per, exp)
    tm.assert_equal(per == base, exp)
    exp = np.array([True, False, True, True])
    exp = tm.box_expected(exp, xbox)
    tm.assert_equal(base != per, exp)
    tm.assert_equal(per != base, exp)
    exp = np.array([False, False, True, True])
    exp = tm.box_expected(exp, xbox)
    tm.assert_equal(base > per, exp)
    tm.assert_equal(per < base, exp)
    exp = np.array([True, False, False, False])
    exp = tm.box_expected(exp, xbox)
    tm.assert_equal(base < per, exp)
    tm.assert_equal(per > base, exp)
    exp = np.array([False, True, True, True])
    exp = tm.box_expected(exp, xbox)
    tm.assert_equal(base >= per, exp)
    tm.assert_equal(per <= base, exp)
    exp = np.array([True, True, False, False])
    exp = tm.box_expected(exp, xbox)
    tm.assert_equal(base <= per, exp)
    tm.assert_equal(per >= base, exp)