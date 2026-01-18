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
@pytest.mark.parametrize('func', [lambda x: x + 1, lambda x: 5], ids=['add', 'constant'])
def test_series_operators_compare(self, comparison_op, func):
    op = comparison_op
    series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
    other = func(series)
    compare_op(series, other, op)