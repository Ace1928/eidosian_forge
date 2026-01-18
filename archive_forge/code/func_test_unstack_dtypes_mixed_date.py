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
@pytest.mark.parametrize('c, d', ((np.zeros(5), np.zeros(5)), (np.arange(5, dtype='f8'), np.arange(5, 10, dtype='f8'))))
def test_unstack_dtypes_mixed_date(self, c, d):
    df = DataFrame({'A': ['a'] * 5, 'C': c, 'D': d, 'B': date_range('2012-01-01', periods=5)})
    right = df.iloc[:3].copy(deep=True)
    df = df.set_index(['A', 'B'])
    df['D'] = df['D'].astype('int64')
    left = df.iloc[:3].unstack(0)
    right = right.set_index(['A', 'B']).unstack(0)
    right['D', 'a'] = right['D', 'a'].astype('int64')
    assert left.shape == (3, 2)
    tm.assert_frame_equal(left, right)