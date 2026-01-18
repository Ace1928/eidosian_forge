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
def test_unstack_with_level_has_nan(self):
    df1 = DataFrame({'L1': [1, 2, 3, 4], 'L2': [3, 4, 1, 2], 'L3': [1, 1, 1, 1], 'x': [1, 2, 3, 4]})
    df1 = df1.set_index(['L1', 'L2', 'L3'])
    new_levels = ['n1', 'n2', 'n3', None]
    df1.index = df1.index.set_levels(levels=new_levels, level='L1')
    df1.index = df1.index.set_levels(levels=new_levels, level='L2')
    result = df1.unstack('L3')['x', 1].sort_index().index
    expected = MultiIndex(levels=[['n1', 'n2', 'n3', None], ['n1', 'n2', 'n3', None]], codes=[[0, 1, 2, 3], [2, 3, 0, 1]], names=['L1', 'L2'])
    tm.assert_index_equal(result, expected)