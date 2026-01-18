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
def test_setitem_with_sparse_value(self):
    df = DataFrame({'c_1': ['a', 'b', 'c'], 'n_1': [1.0, 2.0, 3.0]})
    sp_array = SparseArray([0, 0, 1])
    df['new_column'] = sp_array
    expected = Series(sp_array, name='new_column')
    tm.assert_series_equal(df['new_column'], expected)