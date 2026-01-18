from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reset_index_datetime2(self, tz_naive_fixture):
    tz = tz_naive_fixture
    idx1 = date_range('1/1/2011', periods=5, freq='D', tz=tz, name='idx1')
    idx2 = Index(range(5), name='idx2', dtype='int64')
    idx3 = date_range('1/1/2012', periods=5, freq='MS', tz='Europe/Paris', name='idx3')
    idx = MultiIndex.from_arrays([idx1, idx2, idx3])
    df = DataFrame({'a': np.arange(5, dtype='int64'), 'b': ['A', 'B', 'C', 'D', 'E']}, index=idx)
    expected = DataFrame({'idx1': idx1, 'idx2': np.arange(5, dtype='int64'), 'idx3': idx3, 'a': np.arange(5, dtype='int64'), 'b': ['A', 'B', 'C', 'D', 'E']}, columns=['idx1', 'idx2', 'idx3', 'a', 'b'])
    result = df.reset_index()
    tm.assert_frame_equal(result, expected)