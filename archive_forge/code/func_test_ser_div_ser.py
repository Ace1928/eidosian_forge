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
@pytest.mark.parametrize('dtype1', [np.int64, np.float64, np.uint64])
def test_ser_div_ser(self, switch_numexpr_min_elements, dtype1, any_real_numpy_dtype):
    dtype2 = any_real_numpy_dtype
    first = Series([3, 4, 5, 8], name='first').astype(dtype1)
    second = Series([0, 0, 0, 3], name='second').astype(dtype2)
    with np.errstate(all='ignore'):
        expected = Series(first.values.astype(np.float64) / second.values, dtype='float64', name=None)
    expected.iloc[0:3] = np.inf
    if first.dtype == 'int64' and second.dtype == 'float32':
        if expr.USE_NUMEXPR and switch_numexpr_min_elements == 0:
            expected = expected.astype('float32')
    result = first / second
    tm.assert_series_equal(result, expected)
    assert not result.equals(second / first)