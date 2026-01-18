from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
def test_setitem_frame_dup_cols_dtype(self):
    df = DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=['a', 'b', 'a', 'c'])
    rhs = DataFrame([[0, 1.5], [2, 2.5]], columns=['a', 'a'])
    df['a'] = rhs
    expected = DataFrame([[0, 2, 1.5, 4], [2, 5, 2.5, 7]], columns=['a', 'b', 'a', 'c'])
    tm.assert_frame_equal(df, expected)
    df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'a', 'b'])
    rhs = DataFrame([[0, 1.5], [2, 2.5]], columns=['a', 'a'])
    df['a'] = rhs
    expected = DataFrame([[0, 1.5, 3], [2, 2.5, 6]], columns=['a', 'a', 'b'])
    tm.assert_frame_equal(df, expected)