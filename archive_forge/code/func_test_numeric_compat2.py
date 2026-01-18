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
def test_numeric_compat2(self):
    idx = RangeIndex(0, 10, 2)
    result = idx * 2
    expected = RangeIndex(0, 20, 4)
    tm.assert_index_equal(result, expected, exact=True)
    result = idx + 2
    expected = RangeIndex(2, 12, 2)
    tm.assert_index_equal(result, expected, exact=True)
    result = idx - 2
    expected = RangeIndex(-2, 8, 2)
    tm.assert_index_equal(result, expected, exact=True)
    result = idx / 2
    expected = RangeIndex(0, 5, 1).astype('float64')
    tm.assert_index_equal(result, expected, exact=True)
    result = idx / 4
    expected = RangeIndex(0, 10, 2) / 4
    tm.assert_index_equal(result, expected, exact=True)
    result = idx // 1
    expected = idx
    tm.assert_index_equal(result, expected, exact=True)
    result = idx * idx
    expected = Index(idx.values * idx.values)
    tm.assert_index_equal(result, expected, exact=True)
    idx = RangeIndex(0, 1000, 2)
    result = idx ** 2
    expected = Index(idx._values) ** 2
    tm.assert_index_equal(Index(result.values), expected, exact=True)