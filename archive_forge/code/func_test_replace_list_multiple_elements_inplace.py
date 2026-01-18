import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_replace_list_multiple_elements_inplace(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3]})
    arr = get_array(df, 'a')
    df.replace([1, 2], 4, inplace=True)
    if using_copy_on_write:
        assert np.shares_memory(arr, get_array(df, 'a'))
        assert df._mgr._has_no_reference(0)
    else:
        assert np.shares_memory(arr, get_array(df, 'a'))