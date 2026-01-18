import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import Term
def test_select_with_dups(setup_path):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=['A', 'A', 'B', 'B'])
    df.index = date_range('20130101 9:30', periods=10, freq='min')
    with ensure_clean_store(setup_path) as store:
        store.append('df', df)
        result = store.select('df')
        expected = df
        tm.assert_frame_equal(result, expected, by_blocks=True)
        result = store.select('df', columns=df.columns)
        expected = df
        tm.assert_frame_equal(result, expected, by_blocks=True)
        result = store.select('df', columns=['A'])
        expected = df.loc[:, ['A']]
        tm.assert_frame_equal(result, expected)
    df = concat([DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=['A', 'A', 'B', 'B']), DataFrame(np.random.default_rng(2).integers(0, 10, size=20).reshape(10, 2), columns=['A', 'C'])], axis=1)
    df.index = date_range('20130101 9:30', periods=10, freq='min')
    with ensure_clean_store(setup_path) as store:
        store.append('df', df)
        result = store.select('df')
        expected = df
        tm.assert_frame_equal(result, expected, by_blocks=True)
        result = store.select('df', columns=df.columns)
        expected = df
        tm.assert_frame_equal(result, expected, by_blocks=True)
        expected = df.loc[:, ['A']]
        result = store.select('df', columns=['A'])
        tm.assert_frame_equal(result, expected, by_blocks=True)
        expected = df.loc[:, ['B', 'A']]
        result = store.select('df', columns=['B', 'A'])
        tm.assert_frame_equal(result, expected, by_blocks=True)
    with ensure_clean_store(setup_path) as store:
        store.append('df', df)
        store.append('df', df)
        expected = df.loc[:, ['B', 'A']]
        expected = concat([expected, expected])
        result = store.select('df', columns=['B', 'A'])
        tm.assert_frame_equal(result, expected, by_blocks=True)