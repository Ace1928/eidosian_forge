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
@pytest.mark.parametrize('holder', [Index, Series])
@pytest.mark.parametrize('dtype', [np.int64, np.uint64, np.float64])
def test_ufunc_coercions(self, holder, dtype):
    idx = holder([1, 2, 3, 4, 5], dtype=dtype, name='x')
    box = Series if holder is Series else Index
    result = np.sqrt(idx)
    assert result.dtype == 'f8' and isinstance(result, box)
    exp = Index(np.sqrt(np.array([1, 2, 3, 4, 5], dtype=np.float64)), name='x')
    exp = tm.box_expected(exp, box)
    tm.assert_equal(result, exp)
    result = np.divide(idx, 2.0)
    assert result.dtype == 'f8' and isinstance(result, box)
    exp = Index([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64, name='x')
    exp = tm.box_expected(exp, box)
    tm.assert_equal(result, exp)
    result = idx + 2.0
    assert result.dtype == 'f8' and isinstance(result, box)
    exp = Index([3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64, name='x')
    exp = tm.box_expected(exp, box)
    tm.assert_equal(result, exp)
    result = idx - 2.0
    assert result.dtype == 'f8' and isinstance(result, box)
    exp = Index([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float64, name='x')
    exp = tm.box_expected(exp, box)
    tm.assert_equal(result, exp)
    result = idx * 1.0
    assert result.dtype == 'f8' and isinstance(result, box)
    exp = Index([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64, name='x')
    exp = tm.box_expected(exp, box)
    tm.assert_equal(result, exp)
    result = idx / 2.0
    assert result.dtype == 'f8' and isinstance(result, box)
    exp = Index([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64, name='x')
    exp = tm.box_expected(exp, box)
    tm.assert_equal(result, exp)