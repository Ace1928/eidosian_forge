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
def test_pivot_with_non_observable_dropna(self, dropna):
    df = DataFrame({'A': Categorical([np.nan, 'low', 'high', 'low', 'high'], categories=['low', 'high'], ordered=True), 'B': [0.0, 1.0, 2.0, 3.0, 4.0]})
    msg = 'The default value of observed=False is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.pivot_table(index='A', values='B', dropna=dropna)
    if dropna:
        values = [2.0, 3.0]
        codes = [0, 1]
    else:
        values = [2.0, 3.0, 0.0]
        codes = [0, 1, -1]
    expected = DataFrame({'B': values}, index=Index(Categorical.from_codes(codes, categories=['low', 'high'], ordered=dropna), name='A'))
    tm.assert_frame_equal(result, expected)