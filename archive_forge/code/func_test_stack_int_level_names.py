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
def test_stack_int_level_names(self, future_stack):
    columns = MultiIndex.from_tuples([('A', 'cat', 'long'), ('B', 'cat', 'long'), ('A', 'dog', 'short'), ('B', 'dog', 'short')], names=['exp', 'animal', 'hair_length'])
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), columns=columns)
    exp_animal_stacked = df.stack(level=['exp', 'animal'], future_stack=future_stack)
    animal_hair_stacked = df.stack(level=['animal', 'hair_length'], future_stack=future_stack)
    exp_hair_stacked = df.stack(level=['exp', 'hair_length'], future_stack=future_stack)
    df2 = df.copy()
    df2.columns.names = [0, 1, 2]
    tm.assert_frame_equal(df2.stack(level=[1, 2], future_stack=future_stack), animal_hair_stacked, check_names=False)
    tm.assert_frame_equal(df2.stack(level=[0, 1], future_stack=future_stack), exp_animal_stacked, check_names=False)
    tm.assert_frame_equal(df2.stack(level=[0, 2], future_stack=future_stack), exp_hair_stacked, check_names=False)
    df3 = df.copy()
    df3.columns.names = [2, 0, 1]
    tm.assert_frame_equal(df3.stack(level=[0, 1], future_stack=future_stack), animal_hair_stacked, check_names=False)
    tm.assert_frame_equal(df3.stack(level=[2, 0], future_stack=future_stack), exp_animal_stacked, check_names=False)
    tm.assert_frame_equal(df3.stack(level=[2, 1], future_stack=future_stack), exp_hair_stacked, check_names=False)