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
@pytest.mark.parametrize('consolidate', [True, False])
def test_setitem_partial_column_inplace(self, consolidate, using_array_manager, using_copy_on_write):
    df = DataFrame({'x': [1.1, 2.1, 3.1, 4.1], 'y': [5.1, 6.1, 7.1, 8.1]}, index=[0, 1, 2, 3])
    df.insert(2, 'z', np.nan)
    if not using_array_manager:
        if consolidate:
            df._consolidate_inplace()
            assert len(df._mgr.blocks) == 1
        else:
            assert len(df._mgr.blocks) == 2
    zvals = df['z']._values
    df.loc[2:, 'z'] = 42
    expected = Series([np.nan, np.nan, 42, 42], index=df.index, name='z')
    tm.assert_series_equal(df['z'], expected)
    if not using_copy_on_write:
        tm.assert_numpy_array_equal(zvals, expected.values)
        assert np.shares_memory(zvals, df['z']._values)