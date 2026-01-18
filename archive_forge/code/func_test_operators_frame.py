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
def test_operators_frame(self):
    ts = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
    ts.name = 'ts'
    df = pd.DataFrame({'A': ts})
    tm.assert_series_equal(ts + ts, ts + df['A'], check_names=False)
    tm.assert_series_equal(ts ** ts, ts ** df['A'], check_names=False)
    tm.assert_series_equal(ts < ts, ts < df['A'], check_names=False)
    tm.assert_series_equal(ts / ts, ts / df['A'], check_names=False)