from collections import deque
from datetime import (
from enum import Enum
import functools
import operator
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import (
@pytest.mark.parametrize('op', [operator.add, operator.sub, operator.mul, operator.truediv])
def test_operators_none_as_na(self, op):
    df = DataFrame({'col1': [2, 5.0, 123, None], 'col2': [1, 2, 3, 4]}, dtype=object)
    msg = 'Downcasting object dtype arrays'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        filled = df.fillna(np.nan)
    result = op(df, 3)
    expected = op(filled, 3).astype(object)
    expected[pd.isna(expected)] = np.nan
    tm.assert_frame_equal(result, expected)
    result = op(df, df)
    expected = op(filled, filled).astype(object)
    expected[pd.isna(expected)] = np.nan
    tm.assert_frame_equal(result, expected)
    msg = 'Downcasting object dtype arrays'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = op(df, df.fillna(7))
    tm.assert_frame_equal(result, expected)
    msg = 'Downcasting object dtype arrays'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = op(df.fillna(7), df)
    tm.assert_frame_equal(result, expected)