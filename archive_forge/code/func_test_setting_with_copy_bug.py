from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setting_with_copy_bug(self, using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'a': list(range(4)), 'b': list('ab..'), 'c': ['a', 'b', np.nan, 'd']})
    df_original = df.copy()
    mask = pd.isna(df.c)
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            df[['c']][mask] = df[['b']][mask]
        tm.assert_frame_equal(df, df_original)
    elif warn_copy_on_write:
        with tm.raises_chained_assignment_error():
            df[['c']][mask] = df[['b']][mask]
    else:
        with pytest.raises(SettingWithCopyError, match=msg):
            df[['c']][mask] = df[['b']][mask]