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
def test_stack_nan_level(self, future_stack):
    df_nan = DataFrame(np.arange(4).reshape(2, 2), columns=MultiIndex.from_tuples([('A', np.nan), ('B', 'b')], names=['Upper', 'Lower']), index=Index([0, 1], name='Num'), dtype=np.float64)
    result = df_nan.stack(future_stack=future_stack)
    if future_stack:
        index = MultiIndex(levels=[[0, 1], [np.nan, 'b']], codes=[[0, 0, 1, 1], [0, 1, 0, 1]], names=['Num', 'Lower'])
    else:
        index = MultiIndex.from_tuples([(0, np.nan), (0, 'b'), (1, np.nan), (1, 'b')], names=['Num', 'Lower'])
    expected = DataFrame([[0.0, np.nan], [np.nan, 1], [2.0, np.nan], [np.nan, 3.0]], columns=Index(['A', 'B'], name='Upper'), index=index)
    tm.assert_frame_equal(result, expected)