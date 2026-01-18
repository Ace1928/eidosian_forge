import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@td.skip_array_manager_invalid_test
def test_transpose_get_view_dt64tzget_view(self, using_copy_on_write):
    dti = date_range('2016-01-01', periods=6, tz='US/Pacific')
    arr = dti._data.reshape(3, 2)
    df = DataFrame(arr)
    assert df._mgr.nblocks == 1
    result = df.T
    assert result._mgr.nblocks == 1
    rtrip = result._mgr.blocks[0].values
    if using_copy_on_write:
        assert np.shares_memory(df._mgr.blocks[0].values._ndarray, rtrip._ndarray)
    else:
        assert np.shares_memory(arr._ndarray, rtrip._ndarray)