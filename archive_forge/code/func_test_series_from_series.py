import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('dtype', [None, 'int64'])
def test_series_from_series(dtype, using_copy_on_write, warn_copy_on_write):
    ser = Series([1, 2, 3], name='name')
    result = Series(ser, dtype=dtype)
    assert np.shares_memory(get_array(ser), get_array(result))
    if using_copy_on_write:
        assert result._mgr.blocks[0].refs.has_reference()
    if using_copy_on_write:
        result.iloc[0] = 0
        assert ser.iloc[0] == 1
        assert not np.shares_memory(get_array(ser), get_array(result))
    else:
        with tm.assert_cow_warning(warn_copy_on_write):
            result.iloc[0] = 0
        assert ser.iloc[0] == 0
        assert np.shares_memory(get_array(ser), get_array(result))
    result = Series(ser, dtype=dtype)
    if using_copy_on_write:
        ser.iloc[0] = 0
        assert result.iloc[0] == 1
    else:
        with tm.assert_cow_warning(warn_copy_on_write):
            ser.iloc[0] = 0
        assert result.iloc[0] == 0