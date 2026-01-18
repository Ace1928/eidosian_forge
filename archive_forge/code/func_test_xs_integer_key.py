import re
import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_xs_integer_key(self):
    dates = range(20111201, 20111205)
    ids = list('abcde')
    index = MultiIndex.from_product([dates, ids], names=['date', 'secid'])
    df = DataFrame(np.random.default_rng(2).standard_normal((len(index), 3)), index, ['X', 'Y', 'Z'])
    result = df.xs(20111201, level='date')
    expected = df.loc[20111201, :]
    tm.assert_frame_equal(result, expected)