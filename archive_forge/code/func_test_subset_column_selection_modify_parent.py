import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_subset_column_selection_modify_parent(backend, using_copy_on_write):
    _, DataFrame, _ = backend
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [0.1, 0.2, 0.3]})
    subset = df[['a', 'c']]
    if using_copy_on_write:
        assert np.shares_memory(get_array(subset, 'a'), get_array(df, 'a'))
    df.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(subset, 'a'), get_array(df, 'a'))
    if using_copy_on_write:
        assert np.shares_memory(get_array(subset, 'c'), get_array(df, 'c'))
    expected = DataFrame({'a': [1, 2, 3], 'c': [0.1, 0.2, 0.3]})
    tm.assert_frame_equal(subset, expected)