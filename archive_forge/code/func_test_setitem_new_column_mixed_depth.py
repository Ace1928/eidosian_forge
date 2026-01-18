import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setitem_new_column_mixed_depth(self):
    arrays = [['a', 'top', 'top', 'routine1', 'routine1', 'routine2'], ['', 'OD', 'OD', 'result1', 'result2', 'result1'], ['', 'wx', 'wy', '', '', '']]
    tuples = sorted(zip(*arrays))
    index = MultiIndex.from_tuples(tuples)
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 6)), columns=index)
    result = df.copy()
    expected = df.copy()
    result['b'] = [1, 2, 3, 4]
    expected['b', '', ''] = [1, 2, 3, 4]
    tm.assert_frame_equal(result, expected)