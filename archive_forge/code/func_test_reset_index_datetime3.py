from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reset_index_datetime3(self, tz_naive_fixture):
    tz = tz_naive_fixture
    dti = date_range('20130101', periods=3, tz=tz)
    idx = MultiIndex.from_product([['a', 'b'], dti])
    df = DataFrame(np.arange(6, dtype='int64').reshape(6, 1), columns=['a'], index=idx)
    expected = DataFrame({'level_0': 'a a a b b b'.split(), 'level_1': dti.append(dti), 'a': np.arange(6, dtype='int64')}, columns=['level_0', 'level_1', 'a'])
    result = df.reset_index()
    tm.assert_frame_equal(result, expected)