from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_dict_strings_vs_ints(self):
    df = DataFrame({'Y0': [1, 2], 'Y1': [3, 4]})
    result = df.replace({'replace_string': 'test'})
    tm.assert_frame_equal(result, df)
    result = df['Y0'].replace({'replace_string': 'test'})
    tm.assert_series_equal(result, df['Y0'])