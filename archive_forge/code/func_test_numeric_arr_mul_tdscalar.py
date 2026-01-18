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
@pytest.mark.parametrize('scalar_td', [Timedelta(days=1), Timedelta(days=1).to_timedelta64(), Timedelta(days=1).to_pytimedelta(), Timedelta(days=1).to_timedelta64().astype('timedelta64[s]'), Timedelta(days=1).to_timedelta64().astype('timedelta64[ms]')], ids=lambda x: type(x).__name__)
def test_numeric_arr_mul_tdscalar(self, scalar_td, numeric_idx, box_with_array):
    box = box_with_array
    index = numeric_idx
    expected = TimedeltaIndex([Timedelta(days=n) for n in range(len(index))])
    if isinstance(scalar_td, np.timedelta64):
        dtype = scalar_td.dtype
        expected = expected.astype(dtype)
    elif type(scalar_td) is timedelta:
        expected = expected.astype('m8[us]')
    index = tm.box_expected(index, box)
    expected = tm.box_expected(expected, box)
    result = index * scalar_td
    tm.assert_equal(result, expected)
    commute = scalar_td * index
    tm.assert_equal(commute, expected)