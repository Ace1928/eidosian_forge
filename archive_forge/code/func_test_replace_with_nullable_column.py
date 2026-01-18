from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['boolean', 'Int64', 'Float64'])
def test_replace_with_nullable_column(self, dtype):
    nullable_ser = Series([1, 0, 1], dtype=dtype)
    df = DataFrame({'A': ['A', 'B', 'x'], 'B': nullable_ser})
    result = df.replace('x', 'X')
    expected = DataFrame({'A': ['A', 'B', 'X'], 'B': nullable_ser})
    tm.assert_frame_equal(result, expected)