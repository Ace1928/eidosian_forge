from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_with_None_keeps_categorical(self):
    cat_series = Series(['b', 'b', 'b', 'd'], dtype='category')
    df = DataFrame({'id': Series([5, 4, 3, 2], dtype='float64'), 'col': cat_series})
    result = df.replace({3: None})
    expected = DataFrame({'id': Series([5.0, 4.0, None, 2.0], dtype='object'), 'col': cat_series})
    tm.assert_frame_equal(result, expected)