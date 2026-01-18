from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_nullable_int_with_string_doesnt_cast(self):
    df = DataFrame({'a': [1, 2, 3, np.nan], 'b': ['some', 'strings', 'here', 'he']})
    df['a'] = df['a'].astype('Int64')
    res = df.replace('', np.nan)
    tm.assert_series_equal(res['a'], df['a'])