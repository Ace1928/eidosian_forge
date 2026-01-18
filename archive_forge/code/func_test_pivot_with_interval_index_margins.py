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
def test_pivot_with_interval_index_margins(self):
    ordered_cat = pd.IntervalIndex.from_arrays([0, 0, 1, 1], [1, 1, 2, 2])
    df = DataFrame({'A': np.arange(4, 0, -1, dtype=np.intp), 'B': ['a', 'b', 'a', 'b'], 'C': Categorical(ordered_cat, ordered=True).sort_values(ascending=False)})
    msg = 'The default value of observed=False is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        pivot_tab = pivot_table(df, index='C', columns='B', values='A', aggfunc='sum', margins=True)
    result = pivot_tab['All']
    expected = Series([3, 7, 10], index=Index([pd.Interval(0, 1), pd.Interval(1, 2), 'All'], name='C'), name='All', dtype=np.intp)
    tm.assert_series_equal(result, expected)