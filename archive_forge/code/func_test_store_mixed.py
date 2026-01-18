import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
@pytest.mark.parametrize('compression', [False, pytest.param(True, marks=td.skip_if_windows)])
def test_store_mixed(compression, setup_path):

    def _make_one():
        df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
        df['obj1'] = 'foo'
        df['obj2'] = 'bar'
        df['bool1'] = df['A'] > 0
        df['bool2'] = df['B'] > 0
        df['int1'] = 1
        df['int2'] = 2
        return df._consolidate()
    df1 = _make_one()
    df2 = _make_one()
    _check_roundtrip(df1, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(df2, tm.assert_frame_equal, path=setup_path)
    with ensure_clean_store(setup_path) as store:
        store['obj'] = df1
        tm.assert_frame_equal(store['obj'], df1)
        store['obj'] = df2
        tm.assert_frame_equal(store['obj'], df2)
    _check_roundtrip(df1['obj1'], tm.assert_series_equal, path=setup_path, compression=compression)
    _check_roundtrip(df1['bool1'], tm.assert_series_equal, path=setup_path, compression=compression)
    _check_roundtrip(df1['int1'], tm.assert_series_equal, path=setup_path, compression=compression)