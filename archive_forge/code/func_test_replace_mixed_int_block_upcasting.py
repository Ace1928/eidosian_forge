from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_mixed_int_block_upcasting(self):
    df = DataFrame({'A': Series([1.0, 2.0], dtype='float64'), 'B': Series([0, 1], dtype='int64')})
    expected = DataFrame({'A': Series([1.0, 2.0], dtype='float64'), 'B': Series([0.5, 1], dtype='float64')})
    result = df.replace(0, 0.5)
    tm.assert_frame_equal(result, expected)
    return_value = df.replace(0, 0.5, inplace=True)
    assert return_value is None
    tm.assert_frame_equal(df, expected)