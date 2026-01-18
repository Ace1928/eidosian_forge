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
def test_stack_mixed_dtype(self, multiindex_dataframe_random_data, future_stack):
    frame = multiindex_dataframe_random_data
    df = frame.T
    df['foo', 'four'] = 'foo'
    df = df.sort_index(level=1, axis=1)
    stacked = df.stack(future_stack=future_stack)
    result = df['foo'].stack(future_stack=future_stack).sort_index()
    tm.assert_series_equal(stacked['foo'], result, check_names=False)
    assert result.name is None
    assert stacked['bar'].dtype == np.float64