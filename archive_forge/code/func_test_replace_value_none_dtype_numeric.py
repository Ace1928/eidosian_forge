from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('val', [2, np.nan, 2.0])
def test_replace_value_none_dtype_numeric(self, val):
    df = DataFrame({'a': [1, val]})
    result = df.replace(val, None)
    expected = DataFrame({'a': [1, None]}, dtype=object)
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'a': [1, val]})
    result = df.replace({val: None})
    tm.assert_frame_equal(result, expected)