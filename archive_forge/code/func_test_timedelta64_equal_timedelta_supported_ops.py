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
@pytest.mark.parametrize('op', [operator.add, operator.sub])
def test_timedelta64_equal_timedelta_supported_ops(self, op, box_with_array):
    ser = Series([Timestamp('20130301'), Timestamp('20130228 23:00:00'), Timestamp('20130228 22:00:00'), Timestamp('20130228 21:00:00')])
    obj = box_with_array(ser)
    intervals = ['D', 'h', 'm', 's', 'us']

    def timedelta64(*args):
        return np.sum(list(starmap(np.timedelta64, zip(args, intervals))))
    for d, h, m, s, us in product(*[range(2)] * 5):
        nptd = timedelta64(d, h, m, s, us)
        pytd = timedelta(days=d, hours=h, minutes=m, seconds=s, microseconds=us)
        lhs = op(obj, nptd)
        rhs = op(obj, pytd)
        tm.assert_equal(lhs, rhs)