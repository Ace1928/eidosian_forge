import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_take_fill_value_datetime(self):
    idx = pd.DatetimeIndex(['2011-01-01', '2011-02-01', '2011-03-01'], name='xxx')
    idx = CategoricalIndex(idx)
    result = idx.take(np.array([1, 0, -1]))
    expected = pd.DatetimeIndex(['2011-02-01', '2011-01-01', '2011-03-01'], name='xxx')
    expected = CategoricalIndex(expected)
    tm.assert_index_equal(result, expected)
    result = idx.take(np.array([1, 0, -1]), fill_value=True)
    expected = pd.DatetimeIndex(['2011-02-01', '2011-01-01', 'NaT'], name='xxx')
    exp_cats = pd.DatetimeIndex(['2011-01-01', '2011-02-01', '2011-03-01'])
    expected = CategoricalIndex(expected, categories=exp_cats)
    tm.assert_index_equal(result, expected)
    result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
    expected = pd.DatetimeIndex(['2011-02-01', '2011-01-01', '2011-03-01'], name='xxx')
    expected = CategoricalIndex(expected)
    tm.assert_index_equal(result, expected)
    msg = 'When allow_fill=True and fill_value is not None, all indices must be >= -1'
    with pytest.raises(ValueError, match=msg):
        idx.take(np.array([1, 0, -2]), fill_value=True)
    with pytest.raises(ValueError, match=msg):
        idx.take(np.array([1, 0, -5]), fill_value=True)
    msg = 'index -5 is out of bounds for (axis 0 with )?size 3'
    with pytest.raises(IndexError, match=msg):
        idx.take(np.array([1, -5]))