import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import Term
def test_select_iterator_complete_8014(setup_path):
    chunksize = 10000.0
    with ensure_clean_store(setup_path) as store:
        expected = DataFrame(np.random.default_rng(2).standard_normal((100064, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=100064, freq='s'))
        _maybe_remove(store, 'df')
        store.append('df', expected)
        beg_dt = expected.index[0]
        end_dt = expected.index[-1]
        result = store.select('df')
        tm.assert_frame_equal(expected, result)
        where = f"index >= '{beg_dt}'"
        result = store.select('df', where=where)
        tm.assert_frame_equal(expected, result)
        where = f"index <= '{end_dt}'"
        result = store.select('df', where=where)
        tm.assert_frame_equal(expected, result)
        where = f"index >= '{beg_dt}' & index <= '{end_dt}'"
        result = store.select('df', where=where)
        tm.assert_frame_equal(expected, result)
    with ensure_clean_store(setup_path) as store:
        expected = DataFrame(np.random.default_rng(2).standard_normal((100064, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=100064, freq='s'))
        _maybe_remove(store, 'df')
        store.append('df', expected)
        beg_dt = expected.index[0]
        end_dt = expected.index[-1]
        results = list(store.select('df', chunksize=chunksize))
        result = concat(results)
        tm.assert_frame_equal(expected, result)
        where = f"index >= '{beg_dt}'"
        results = list(store.select('df', where=where, chunksize=chunksize))
        result = concat(results)
        tm.assert_frame_equal(expected, result)
        where = f"index <= '{end_dt}'"
        results = list(store.select('df', where=where, chunksize=chunksize))
        result = concat(results)
        tm.assert_frame_equal(expected, result)
        where = f"index >= '{beg_dt}' & index <= '{end_dt}'"
        results = list(store.select('df', where=where, chunksize=chunksize))
        result = concat(results)
        tm.assert_frame_equal(expected, result)