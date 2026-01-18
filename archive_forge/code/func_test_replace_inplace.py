from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't set float into string")
def test_replace_inplace(self, datetime_frame, float_string_frame):
    datetime_frame.loc[datetime_frame.index[:5], 'A'] = np.nan
    datetime_frame.loc[datetime_frame.index[-5:], 'A'] = np.nan
    tsframe = datetime_frame.copy()
    return_value = tsframe.replace(np.nan, 0, inplace=True)
    assert return_value is None
    tm.assert_frame_equal(tsframe, datetime_frame.fillna(0))
    mf = float_string_frame
    mf.iloc[5:20, mf.columns.get_loc('foo')] = np.nan
    mf.iloc[-10:, mf.columns.get_loc('A')] = np.nan
    result = float_string_frame.replace(np.nan, 0)
    expected = float_string_frame.fillna(value=0)
    tm.assert_frame_equal(result, expected)
    tsframe = datetime_frame.copy()
    return_value = tsframe.replace([np.nan], [0], inplace=True)
    assert return_value is None
    tm.assert_frame_equal(tsframe, datetime_frame.fillna(0))