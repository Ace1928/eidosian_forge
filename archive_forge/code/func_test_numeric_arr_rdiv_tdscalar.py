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
def test_numeric_arr_rdiv_tdscalar(self, three_days, numeric_idx, box_with_array):
    box = box_with_array
    index = numeric_idx[1:3]
    expected = TimedeltaIndex(['3 Days', '36 Hours'])
    if isinstance(three_days, np.timedelta64):
        dtype = three_days.dtype
        if dtype < np.dtype('m8[s]'):
            dtype = np.dtype('m8[s]')
        expected = expected.astype(dtype)
    elif type(three_days) is timedelta:
        expected = expected.astype('m8[us]')
    elif isinstance(three_days, (pd.offsets.Day, pd.offsets.Hour, pd.offsets.Minute, pd.offsets.Second)):
        expected = expected.astype('m8[s]')
    index = tm.box_expected(index, box)
    expected = tm.box_expected(expected, box)
    result = three_days / index
    tm.assert_equal(result, expected)
    msg = 'cannot use operands with types dtype'
    with pytest.raises(TypeError, match=msg):
        index / three_days