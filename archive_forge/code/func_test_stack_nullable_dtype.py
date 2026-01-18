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
@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_nullable_dtype(self, future_stack):
    columns = MultiIndex.from_product([['54511', '54515'], ['r', 't_mean']], names=['station', 'element'])
    index = Index([1, 2, 3], name='time')
    arr = np.array([[50, 226, 10, 215], [10, 215, 9, 220], [305, 232, 111, 220]])
    df = DataFrame(arr, columns=columns, index=index, dtype=pd.Int64Dtype())
    result = df.stack('station', future_stack=future_stack)
    expected = df.astype(np.int64).stack('station', future_stack=future_stack).astype(pd.Int64Dtype())
    tm.assert_frame_equal(result, expected)
    df[df.columns[0]] = df[df.columns[0]].astype(pd.Float64Dtype())
    result = df.stack('station', future_stack=future_stack)
    expected = DataFrame({'r': pd.array([50.0, 10.0, 10.0, 9.0, 305.0, 111.0], dtype=pd.Float64Dtype()), 't_mean': pd.array([226, 215, 215, 220, 232, 220], dtype=pd.Int64Dtype())}, index=MultiIndex.from_product([index, columns.levels[0]]))
    expected.columns.name = 'element'
    tm.assert_frame_equal(result, expected)