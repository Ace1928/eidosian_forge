import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_replace_list_inplace_refs_categorical(using_copy_on_write):
    df = DataFrame({'a': ['a', 'b', 'c']}, dtype='category')
    view = df[:]
    df_orig = df.copy()
    msg = 'The behavior of Series\\.replace \\(and DataFrame.replace\\) with CategoricalDtype'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.replace(['c'], value='a', inplace=True)
    if using_copy_on_write:
        assert not np.shares_memory(get_array(view, 'a').codes, get_array(df, 'a').codes)
        tm.assert_frame_equal(df_orig, view)
    else:
        assert not np.shares_memory(get_array(view, 'a').codes, get_array(df, 'a').codes)