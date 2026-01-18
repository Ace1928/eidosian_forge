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
def test_setitem_duplicate_columns_not_inplace(self):
    cols = ['A', 'B'] * 2
    df = DataFrame(0.0, index=[0], columns=cols)
    df_copy = df.copy()
    df_view = df[:]
    df['B'] = (2, 5)
    expected = DataFrame([[0.0, 2, 0.0, 5]], columns=cols)
    tm.assert_frame_equal(df_view, df_copy)
    tm.assert_frame_equal(df, expected)