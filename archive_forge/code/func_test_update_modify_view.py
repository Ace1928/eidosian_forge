import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@td.skip_array_manager_invalid_test
def test_update_modify_view(self, using_copy_on_write, warn_copy_on_write, using_infer_string):
    df = DataFrame({'A': ['1', np.nan], 'B': ['100', np.nan]})
    df2 = DataFrame({'A': ['a', 'x'], 'B': ['100', '200']})
    df2_orig = df2.copy()
    result_view = df2[:]
    with tm.assert_cow_warning(warn_copy_on_write):
        df2.update(df)
    expected = DataFrame({'A': ['1', 'x'], 'B': ['100', '200']})
    tm.assert_frame_equal(df2, expected)
    if using_copy_on_write or using_infer_string:
        tm.assert_frame_equal(result_view, df2_orig)
    else:
        tm.assert_frame_equal(result_view, expected)