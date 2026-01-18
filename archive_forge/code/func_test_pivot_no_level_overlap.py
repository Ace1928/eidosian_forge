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
def test_pivot_no_level_overlap(self):
    data = DataFrame({'a': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'] * 2, 'b': [0, 0, 0, 0, 1, 1, 1, 1] * 2, 'c': (['foo'] * 4 + ['bar'] * 4) * 2, 'value': np.random.default_rng(2).standard_normal(16)})
    table = data.pivot_table('value', index='a', columns=['b', 'c'])
    grouped = data.groupby(['a', 'b', 'c'])['value'].mean()
    expected = grouped.unstack('b').unstack('c').dropna(axis=1, how='all')
    tm.assert_frame_equal(table, expected)