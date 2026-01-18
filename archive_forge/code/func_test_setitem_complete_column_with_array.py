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
def test_setitem_complete_column_with_array(self):
    df = DataFrame({'a': ['one', 'two', 'three'], 'b': [1, 2, 3]})
    arr = np.array([[1, 1], [3, 1], [5, 1]])
    df[['c', 'd']] = arr
    expected = DataFrame({'a': ['one', 'two', 'three'], 'b': [1, 2, 3], 'c': [1, 3, 5], 'd': [1, 1, 1]})
    expected['c'] = expected['c'].astype(arr.dtype)
    expected['d'] = expected['d'].astype(arr.dtype)
    assert expected['c'].dtype == arr.dtype
    assert expected['d'].dtype == arr.dtype
    tm.assert_frame_equal(df, expected)