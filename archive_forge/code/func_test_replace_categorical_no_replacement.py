from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_categorical_no_replacement(self):
    df = DataFrame({'a': ['one', 'two', None, 'three'], 'b': ['one', None, 'two', 'three']}, dtype='category')
    expected = df.copy()
    result = df.replace(to_replace=['.', 'def'], value=['_', None])
    tm.assert_frame_equal(result, expected)