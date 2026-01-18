from __future__ import annotations
from collections import abc
from datetime import timedelta
from decimal import Decimal
import operator
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.tests.arithmetic.common import (
def test_series_frame_radd_bug(self, fixed_now_ts):
    vals = Series([str(i) for i in range(5)])
    result = 'foo_' + vals
    expected = vals.map(lambda x: 'foo_' + x)
    tm.assert_series_equal(result, expected)
    frame = pd.DataFrame({'vals': vals})
    result = 'foo_' + frame
    expected = pd.DataFrame({'vals': vals.map(lambda x: 'foo_' + x)})
    tm.assert_frame_equal(result, expected)
    ts = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
    fix_now = fixed_now_ts.to_pydatetime()
    msg = '|'.join(['unsupported operand type', 'Concatenation operation'])
    with pytest.raises(TypeError, match=msg):
        fix_now + ts
    with pytest.raises(TypeError, match=msg):
        ts + fix_now