from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_series_dict(self):
    df = DataFrame({'zero': {'a': 0.0, 'b': 1}, 'one': {'a': 2.0, 'b': 0}})
    result = df.replace(0, {'zero': 0.5, 'one': 1.0})
    expected = DataFrame({'zero': {'a': 0.5, 'b': 1}, 'one': {'a': 2.0, 'b': 1.0}})
    tm.assert_frame_equal(result, expected)
    result = df.replace(0, df.mean())
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'zero': {'a': 0.0, 'b': 1}, 'one': {'a': 2.0, 'b': 0}})
    s = Series({'zero': 0.0, 'one': 2.0})
    result = df.replace(s, {'zero': 0.5, 'one': 1.0})
    expected = DataFrame({'zero': {'a': 0.5, 'b': 1}, 'one': {'a': 1.0, 'b': 0.0}})
    tm.assert_frame_equal(result, expected)
    result = df.replace(s, df.mean())
    tm.assert_frame_equal(result, expected)