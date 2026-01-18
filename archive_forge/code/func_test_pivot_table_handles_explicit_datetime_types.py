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
def test_pivot_table_handles_explicit_datetime_types(self):
    df = DataFrame([{'a': 'x', 'date_str': '2023-01-01', 'amount': 1}, {'a': 'y', 'date_str': '2023-01-02', 'amount': 2}, {'a': 'z', 'date_str': '2023-01-03', 'amount': 3}])
    df['date'] = pd.to_datetime(df['date_str'])
    with tm.assert_produces_warning(False):
        pivot = df.pivot_table(index=['a', 'date'], values=['amount'], aggfunc='sum', margins=True)
    expected = MultiIndex.from_tuples([('x', datetime.strptime('2023-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')), ('y', datetime.strptime('2023-01-02 00:00:00', '%Y-%m-%d %H:%M:%S')), ('z', datetime.strptime('2023-01-03 00:00:00', '%Y-%m-%d %H:%M:%S')), ('All', '')], names=['a', 'date'])
    tm.assert_index_equal(pivot.index, expected)