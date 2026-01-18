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
@pytest.mark.parametrize('left, right, op_fail', [[[Timestamp('20111230'), Timestamp('20120101'), NaT], [Timestamp('20111231'), Timestamp('20120102'), Timestamp('20120104')], ['__sub__', '__rsub__']], [[Timestamp('20111230'), Timestamp('20120101'), NaT], [timedelta(minutes=5, seconds=3), timedelta(minutes=5, seconds=3), NaT], ['__add__', '__radd__', '__sub__']], [[Timestamp('20111230', tz='US/Eastern'), Timestamp('20111230', tz='US/Eastern'), NaT], [timedelta(minutes=5, seconds=3), NaT, timedelta(minutes=5, seconds=3)], ['__add__', '__radd__', '__sub__']]])
def test_operators_datetimelike_invalid(self, left, right, op_fail, all_arithmetic_operators):
    op_str = all_arithmetic_operators
    arg1 = Series(left)
    arg2 = Series(right)
    op = getattr(arg1, op_str, None)
    if op_str not in op_fail:
        with pytest.raises(TypeError, match='operate|[cC]annot|unsupported operand'):
            op(arg2)
    else:
        op(arg2)