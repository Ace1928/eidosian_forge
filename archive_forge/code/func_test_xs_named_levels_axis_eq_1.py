import re
import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
@pytest.mark.parametrize('key, level, exp_arr, exp_index', [('a', 'lvl0', lambda x: x[:, 0:2], Index(['bar', 'foo'], name='lvl1')), ('foo', 'lvl1', lambda x: x[:, 1:2], Index(['a'], name='lvl0'))])
def test_xs_named_levels_axis_eq_1(self, key, level, exp_arr, exp_index):
    arr = np.random.default_rng(2).standard_normal((4, 4))
    index = MultiIndex(levels=[['a', 'b'], ['bar', 'foo', 'hello', 'world']], codes=[[0, 0, 1, 1], [0, 1, 2, 3]], names=['lvl0', 'lvl1'])
    df = DataFrame(arr, columns=index)
    result = df.xs(key, level=level, axis=1)
    expected = DataFrame(exp_arr(arr), columns=exp_index)
    tm.assert_frame_equal(result, expected)