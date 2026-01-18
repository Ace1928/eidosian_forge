import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('value', ['d', None])
def test_replace_object_list_inplace(using_copy_on_write, value):
    df = DataFrame({'a': ['a', 'b', 'c']})
    arr = get_array(df, 'a')
    df.replace(['c'], value, inplace=True)
    if using_copy_on_write or value is None:
        assert np.shares_memory(arr, get_array(df, 'a'))
    else:
        assert not np.shares_memory(arr, get_array(df, 'a'))
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)