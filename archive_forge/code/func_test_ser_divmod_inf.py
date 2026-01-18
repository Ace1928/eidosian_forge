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
def test_ser_divmod_inf(self):
    left = Series([np.inf, 1.0])
    right = Series([np.inf, 2.0])
    expected = (left // right, left % right)
    result = divmod(left, right)
    tm.assert_series_equal(result[0], expected[0])
    tm.assert_series_equal(result[1], expected[1])
    result = divmod(left.values, right)
    tm.assert_series_equal(result[0], expected[0])
    tm.assert_series_equal(result[1], expected[1])