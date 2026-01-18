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
def test_setitem_listlike_views(self, using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 4, 6]})
    ser = df['a']
    df[['c', 'd']] = np.array([[0.1, 0.2], [0.3, 0.4], [0.4, 0.5]])
    with tm.assert_cow_warning(warn_copy_on_write):
        df.iloc[0, 0] = 100
    if using_copy_on_write:
        expected = Series([1, 2, 3], name='a')
    else:
        expected = Series([100, 2, 3], name='a')
    tm.assert_series_equal(ser, expected)