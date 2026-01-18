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
def test_stack_order_with_unsorted_levels_multi_row_2(self, future_stack):
    levels = ((0, 1), (1, 0))
    stack_lev = 1
    columns = MultiIndex(levels=levels, codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
    df = DataFrame(columns=columns, data=[range(4)], index=[1, 0, 2, 3])
    kwargs = {} if future_stack else {'sort': True}
    result = df.stack(stack_lev, future_stack=future_stack, **kwargs)
    expected_index = MultiIndex(levels=[[0, 1, 2, 3], [0, 1]], codes=[[1, 1, 0, 0, 2, 2, 3, 3], [1, 0, 1, 0, 1, 0, 1, 0]])
    expected = DataFrame({0: [0, 1, 0, 1, 0, 1, 0, 1], 1: [2, 3, 2, 3, 2, 3, 2, 3]}, index=expected_index)
    tm.assert_frame_equal(result, expected)