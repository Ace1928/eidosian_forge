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
def test_unstack_preserve_dtypes(self):
    df = DataFrame({'state': ['IL', 'MI', 'NC'], 'index': ['a', 'b', 'c'], 'some_categories': Series(['a', 'b', 'c']).astype('category'), 'A': np.random.default_rng(2).random(3), 'B': 1, 'C': 'foo', 'D': pd.Timestamp('20010102'), 'E': Series([1.0, 50.0, 100.0]).astype('float32'), 'F': Series([3.0, 4.0, 5.0]).astype('float64'), 'G': False, 'H': Series([1, 200, 923442]).astype('int8')})

    def unstack_and_compare(df, column_name):
        unstacked1 = df.unstack([column_name])
        unstacked2 = df.unstack(column_name)
        tm.assert_frame_equal(unstacked1, unstacked2)
    df1 = df.set_index(['state', 'index'])
    unstack_and_compare(df1, 'index')
    df1 = df.set_index(['state', 'some_categories'])
    unstack_and_compare(df1, 'some_categories')
    df1 = df.set_index(['F', 'C'])
    unstack_and_compare(df1, 'F')
    df1 = df.set_index(['G', 'B', 'state'])
    unstack_and_compare(df1, 'B')
    df1 = df.set_index(['E', 'A'])
    unstack_and_compare(df1, 'E')
    df1 = df.set_index(['state', 'index'])
    s = df1['A']
    unstack_and_compare(s, 'index')