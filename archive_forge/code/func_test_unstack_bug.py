from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
def test_unstack_bug(self, future_stack):
    df = DataFrame({'state': ['naive', 'naive', 'naive', 'active', 'active', 'active'], 'exp': ['a', 'b', 'b', 'b', 'a', 'a'], 'barcode': [1, 2, 3, 4, 1, 3], 'v': ['hi', 'hi', 'bye', 'bye', 'bye', 'peace'], 'extra': np.arange(6.0)})
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby(['state', 'exp', 'barcode', 'v']).apply(len)
    unstacked = result.unstack()
    restacked = unstacked.stack(future_stack=future_stack)
    tm.assert_series_equal(restacked, result.reindex(restacked.index).astype(float))