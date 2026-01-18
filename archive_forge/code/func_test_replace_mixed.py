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
def test_replace_mixed(self, float_string_frame):
    mf = float_string_frame
    mf.iloc[5:20, mf.columns.get_loc('foo')] = np.nan
    mf.iloc[-10:, mf.columns.get_loc('A')] = np.nan
    result = float_string_frame.replace(np.nan, -18)
    expected = float_string_frame.fillna(value=-18)
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(result.replace(-18, np.nan), float_string_frame)
    result = float_string_frame.replace(np.nan, -100000000.0)
    expected = float_string_frame.fillna(value=-100000000.0)
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(result.replace(-100000000.0, np.nan), float_string_frame)