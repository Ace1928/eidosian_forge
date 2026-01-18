from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.parametrize('op', [ops.rand_, ops.ror_])
def test_reversed_logical_op_with_index_returns_series(self, op):
    ser = Series([True, True, False, False])
    idx1 = Index([True, False, True, False])
    idx2 = Index([1, 0, 1, 0])
    expected = Series(op(idx1.values, ser.values))
    result = op(ser, idx1)
    tm.assert_series_equal(result, expected)
    expected = op(ser, Series(idx2))
    result = op(ser, idx2)
    tm.assert_series_equal(result, expected)