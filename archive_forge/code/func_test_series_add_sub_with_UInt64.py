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
def test_series_add_sub_with_UInt64():
    series1 = Series([1, 2, 3])
    series2 = Series([2, 1, 3], dtype='UInt64')
    result = series1 + series2
    expected = Series([3, 3, 6], dtype='Float64')
    tm.assert_series_equal(result, expected)
    result = series1 - series2
    expected = Series([-1, 1, 0], dtype='Float64')
    tm.assert_series_equal(result, expected)