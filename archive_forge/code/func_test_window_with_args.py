import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
def test_window_with_args(step):
    pytest.importorskip('scipy')
    r = Series(np.random.default_rng(2).standard_normal(100)).rolling(window=10, min_periods=1, win_type='gaussian', step=step)
    expected = concat([r.mean(std=10), r.mean(std=0.01)], axis=1)
    expected.columns = ['<lambda>', '<lambda>']
    result = r.aggregate([lambda x: x.mean(std=10), lambda x: x.mean(std=0.01)])
    tm.assert_frame_equal(result, expected)

    def a(x):
        return x.mean(std=10)

    def b(x):
        return x.mean(std=0.01)
    expected = concat([r.mean(std=10), r.mean(std=0.01)], axis=1)
    expected.columns = ['a', 'b']
    result = r.aggregate([a, b])
    tm.assert_frame_equal(result, expected)