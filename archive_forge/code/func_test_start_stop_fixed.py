import contextlib
import datetime as dt
import hashlib
import tempfile
import time
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import (
def test_start_stop_fixed(setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame({'A': np.random.default_rng(2).random(20), 'B': np.random.default_rng(2).random(20)}, index=date_range('20130101', periods=20))
        store.put('df', df)
        result = store.select('df', start=0, stop=5)
        expected = df.iloc[0:5, :]
        tm.assert_frame_equal(result, expected)
        result = store.select('df', start=5, stop=10)
        expected = df.iloc[5:10, :]
        tm.assert_frame_equal(result, expected)
        result = store.select('df', start=30, stop=40)
        expected = df.iloc[30:40, :]
        tm.assert_frame_equal(result, expected)
        s = df.A
        store.put('s', s)
        result = store.select('s', start=0, stop=5)
        expected = s.iloc[0:5]
        tm.assert_series_equal(result, expected)
        result = store.select('s', start=5, stop=10)
        expected = s.iloc[5:10]
        tm.assert_series_equal(result, expected)
        df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
        df.iloc[3:5, 1:3] = np.nan
        df.iloc[8:10, -2] = np.nan