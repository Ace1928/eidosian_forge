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
def test_setitem_empty_df_duplicate_columns(self, using_copy_on_write):
    df = DataFrame(columns=['a', 'b', 'b'], dtype='float64')
    df.loc[:, 'a'] = list(range(2))
    expected = DataFrame([[0, np.nan, np.nan], [1, np.nan, np.nan]], columns=['a', 'b', 'b'])
    tm.assert_frame_equal(df, expected)