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
def test_setitem_rhs_dataframe(self):
    df = DataFrame({'a': [1, 2]})
    df['a'] = DataFrame({'a': [10, 11]}, index=[1, 2])
    expected = DataFrame({'a': [np.nan, 10]})
    tm.assert_frame_equal(df, expected)
    df = DataFrame({'a': [1, 2]})
    df.isetitem(0, DataFrame({'a': [10, 11]}, index=[1, 2]))
    tm.assert_frame_equal(df, expected)