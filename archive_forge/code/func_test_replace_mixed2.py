from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_mixed2(self, using_infer_string):
    df = DataFrame({'A': Series([1.0, 2.0], dtype='float64'), 'B': Series([0, 1], dtype='int64')})
    expected = DataFrame({'A': Series([1, 'foo'], dtype='object'), 'B': Series([0, 1], dtype='int64')})
    result = df.replace(2, 'foo')
    tm.assert_frame_equal(result, expected)
    expected = DataFrame({'A': Series(['foo', 'bar']), 'B': Series([0, 'foo'], dtype='object')})
    if using_infer_string:
        with tm.assert_produces_warning(FutureWarning, match='Downcasting'):
            result = df.replace([1, 2], ['foo', 'bar'])
    else:
        result = df.replace([1, 2], ['foo', 'bar'])
    tm.assert_frame_equal(result, expected)