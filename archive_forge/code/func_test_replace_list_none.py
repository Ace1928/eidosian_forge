import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_replace_list_none(using_copy_on_write):
    df = DataFrame({'a': ['a', 'b', 'c']})
    df_orig = df.copy()
    df2 = df.replace(['b'], value=None)
    tm.assert_frame_equal(df, df_orig)
    assert not np.shares_memory(get_array(df, 'a'), get_array(df2, 'a'))