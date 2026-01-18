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
def test_unstack_level_binding(self, future_stack):
    mi = MultiIndex(levels=[['foo', 'bar'], ['one', 'two'], ['a', 'b']], codes=[[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0]], names=['first', 'second', 'third'])
    s = Series(0, index=mi)
    result = s.unstack([1, 2]).stack(0, future_stack=future_stack)
    expected_mi = MultiIndex(levels=[['foo', 'bar'], ['one', 'two']], codes=[[0, 0, 1, 1], [0, 1, 0, 1]], names=['first', 'second'])
    expected = DataFrame(np.array([[0, np.nan], [np.nan, 0], [0, np.nan], [np.nan, 0]], dtype=np.float64), index=expected_mi, columns=Index(['b', 'a'], name='third'))
    tm.assert_frame_equal(result, expected)