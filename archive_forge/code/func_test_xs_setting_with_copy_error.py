import re
import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_xs_setting_with_copy_error(self, multiindex_dataframe_random_data, using_copy_on_write, warn_copy_on_write):
    df = multiindex_dataframe_random_data
    df_orig = df.copy()
    result = df.xs('two', level='second')
    if using_copy_on_write or warn_copy_on_write:
        result[:] = 10
    else:
        msg = 'A value is trying to be set on a copy of a slice from a DataFrame'
        with pytest.raises(SettingWithCopyError, match=msg):
            result[:] = 10
    tm.assert_frame_equal(df, df_orig)