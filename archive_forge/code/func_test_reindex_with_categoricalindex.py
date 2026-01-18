from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
def test_reindex_with_categoricalindex(self):
    df = DataFrame({'A': np.arange(3, dtype='int64')}, index=CategoricalIndex(list('abc'), dtype=CategoricalDtype(list('cabe')), name='B'))
    result = df.reindex(['a', 'b', 'e'])
    expected = DataFrame({'A': [0, 1, np.nan], 'B': Series(list('abe'))}).set_index('B')
    tm.assert_frame_equal(result, expected, check_index_type=True)
    result = df.reindex(['a', 'b'])
    expected = DataFrame({'A': [0, 1], 'B': Series(list('ab'))}).set_index('B')
    tm.assert_frame_equal(result, expected, check_index_type=True)
    result = df.reindex(['e'])
    expected = DataFrame({'A': [np.nan], 'B': Series(['e'])}).set_index('B')
    tm.assert_frame_equal(result, expected, check_index_type=True)
    result = df.reindex(['d'])
    expected = DataFrame({'A': [np.nan], 'B': Series(['d'])}).set_index('B')
    tm.assert_frame_equal(result, expected, check_index_type=True)
    cats = list('cabe')
    result = df.reindex(Categorical(['a', 'e'], categories=cats))
    expected = DataFrame({'A': [0, np.nan], 'B': Series(list('ae')).astype(CategoricalDtype(cats))}).set_index('B')
    tm.assert_frame_equal(result, expected, check_index_type=True)
    result = df.reindex(Categorical(['a'], categories=cats))
    expected = DataFrame({'A': [0], 'B': Series(list('a')).astype(CategoricalDtype(cats))}).set_index('B')
    tm.assert_frame_equal(result, expected, check_index_type=True)
    result = df.reindex(['a', 'b', 'e'])
    expected = DataFrame({'A': [0, 1, np.nan], 'B': Series(list('abe'))}).set_index('B')
    tm.assert_frame_equal(result, expected, check_index_type=True)
    result = df.reindex(['a', 'b'])
    expected = DataFrame({'A': [0, 1], 'B': Series(list('ab'))}).set_index('B')
    tm.assert_frame_equal(result, expected, check_index_type=True)
    result = df.reindex(['e'])
    expected = DataFrame({'A': [np.nan], 'B': Series(['e'])}).set_index('B')
    tm.assert_frame_equal(result, expected, check_index_type=True)
    result = df.reindex(Categorical(['a', 'e'], categories=cats, ordered=True))
    expected = DataFrame({'A': [0, np.nan], 'B': Series(list('ae')).astype(CategoricalDtype(cats, ordered=True))}).set_index('B')
    tm.assert_frame_equal(result, expected, check_index_type=True)
    result = df.reindex(Categorical(['a', 'd'], categories=['a', 'd']))
    expected = DataFrame({'A': [0, np.nan], 'B': Series(list('ad')).astype(CategoricalDtype(['a', 'd']))}).set_index('B')
    tm.assert_frame_equal(result, expected, check_index_type=True)
    df2 = DataFrame({'A': np.arange(6, dtype='int64')}, index=CategoricalIndex(list('aabbca'), dtype=CategoricalDtype(list('cabe')), name='B'))
    msg = 'cannot reindex on an axis with duplicate labels'
    with pytest.raises(ValueError, match=msg):
        df2.reindex(['a', 'b'])
    msg = 'argument {} is not implemented for CategoricalIndex\\.reindex'
    with pytest.raises(NotImplementedError, match=msg.format('method')):
        df.reindex(['a'], method='ffill')
    with pytest.raises(NotImplementedError, match=msg.format('level')):
        df.reindex(['a'], level=1)
    with pytest.raises(NotImplementedError, match=msg.format('limit')):
        df.reindex(['a'], limit=2)