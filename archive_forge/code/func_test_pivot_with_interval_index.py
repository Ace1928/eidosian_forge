from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
def test_pivot_with_interval_index(self, interval_values, dropna):
    df = DataFrame({'A': interval_values, 'B': 1})
    msg = 'The default value of observed=False is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.pivot_table(index='A', values='B', dropna=dropna)
    expected = DataFrame({'B': 1.0}, index=Index(interval_values.unique(), name='A'))
    if not dropna:
        expected = expected.astype(float)
    tm.assert_frame_equal(result, expected)