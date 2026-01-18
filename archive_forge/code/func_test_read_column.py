from contextlib import closing
from pathlib import Path
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
from pandas.io.pytables import TableIterator
def test_read_column(setup_path):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
    with ensure_clean_store(setup_path) as store:
        _maybe_remove(store, 'df')
        with pytest.raises(KeyError, match='No object named df in the file'):
            store.select_column('df', 'index')
        store.append('df', df)
        with pytest.raises(KeyError, match=re.escape("'column [foo] not found in the table'")):
            store.select_column('df', 'foo')
        msg = re.escape("select_column() got an unexpected keyword argument 'where'")
        with pytest.raises(TypeError, match=msg):
            store.select_column('df', 'index', where=['index>5'])
        result = store.select_column('df', 'index')
        tm.assert_almost_equal(result.values, Series(df.index).values)
        assert isinstance(result, Series)
        msg = re.escape('column [values_block_0] can not be extracted individually; it is not data indexable')
        with pytest.raises(ValueError, match=msg):
            store.select_column('df', 'values_block_0')
        df2 = df.copy()
        df2['string'] = 'foo'
        store.append('df2', df2, data_columns=['string'])
        result = store.select_column('df2', 'string')
        tm.assert_almost_equal(result.values, df2['string'].values)
        df3 = df.copy()
        df3['string'] = 'foo'
        df3.loc[df3.index[4:6], 'string'] = np.nan
        store.append('df3', df3, data_columns=['string'])
        result = store.select_column('df3', 'string')
        tm.assert_almost_equal(result.values, df3['string'].values)
        result = store.select_column('df3', 'string', start=2)
        tm.assert_almost_equal(result.values, df3['string'].values[2:])
        result = store.select_column('df3', 'string', start=-2)
        tm.assert_almost_equal(result.values, df3['string'].values[-2:])
        result = store.select_column('df3', 'string', stop=2)
        tm.assert_almost_equal(result.values, df3['string'].values[:2])
        result = store.select_column('df3', 'string', stop=-2)
        tm.assert_almost_equal(result.values, df3['string'].values[:-2])
        result = store.select_column('df3', 'string', start=2, stop=-2)
        tm.assert_almost_equal(result.values, df3['string'].values[2:-2])
        result = store.select_column('df3', 'string', start=-2, stop=2)
        tm.assert_almost_equal(result.values, df3['string'].values[-2:2])
        df4 = DataFrame({'A': np.random.default_rng(2).standard_normal(10), 'B': 'foo'})
        store.append('df4', df4, data_columns=True)
        expected = df4['B']
        result = store.select_column('df4', 'B')
        tm.assert_series_equal(result, expected)