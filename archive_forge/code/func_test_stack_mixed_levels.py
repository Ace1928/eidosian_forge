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
def test_stack_mixed_levels(self, future_stack):
    columns = MultiIndex.from_tuples([('A', 'cat', 'long'), ('B', 'cat', 'long'), ('A', 'dog', 'short'), ('B', 'dog', 'short')], names=['exp', 'animal', 'hair_length'])
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), columns=columns)
    animal_hair_stacked = df.stack(level=['animal', 'hair_length'], future_stack=future_stack)
    exp_hair_stacked = df.stack(level=['exp', 'hair_length'], future_stack=future_stack)
    df2 = df.copy()
    df2.columns.names = ['exp', 'animal', 1]
    tm.assert_frame_equal(df2.stack(level=['animal', 1], future_stack=future_stack), animal_hair_stacked, check_names=False)
    tm.assert_frame_equal(df2.stack(level=['exp', 1], future_stack=future_stack), exp_hair_stacked, check_names=False)
    msg = 'level should contain all level names or all level numbers, not a mixture of the two'
    with pytest.raises(ValueError, match=msg):
        df2.stack(level=['animal', 0], future_stack=future_stack)
    df3 = df.copy()
    df3.columns.names = ['exp', 'animal', 0]
    tm.assert_frame_equal(df3.stack(level=['animal', 0], future_stack=future_stack), animal_hair_stacked, check_names=False)