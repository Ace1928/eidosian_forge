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
@pytest.mark.parametrize('holder', [Index, RangeIndex, Series])
@pytest.mark.parametrize('dtype', [np.int64, np.uint64, np.float64])
def test_ufunc_compat(self, holder, dtype):
    box = Series if holder is Series else Index
    if holder is RangeIndex:
        if dtype != np.int64:
            pytest.skip(f'dtype {dtype} not relevant for RangeIndex')
        idx = RangeIndex(0, 5, name='foo')
    else:
        idx = holder(np.arange(5, dtype=dtype), name='foo')
    result = np.sin(idx)
    expected = box(np.sin(np.arange(5, dtype=dtype)), name='foo')
    tm.assert_equal(result, expected)