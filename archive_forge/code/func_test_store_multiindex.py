import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
def test_store_multiindex(setup_path):
    with ensure_clean_store(setup_path) as store:

        def make_index(names=None):
            dti = date_range('2013-12-01', '2013-12-02')
            mi = MultiIndex.from_product([dti, range(2), range(3)], names=names)
            return mi
        _maybe_remove(store, 'df')
        df = DataFrame(np.zeros((12, 2)), columns=['a', 'b'], index=make_index())
        store.append('df', df)
        tm.assert_frame_equal(store.select('df'), df)
        _maybe_remove(store, 'df')
        df = DataFrame(np.zeros((12, 2)), columns=['a', 'b'], index=make_index(['date', None, None]))
        store.append('df', df)
        tm.assert_frame_equal(store.select('df'), df)
        _maybe_remove(store, 'ser')
        ser = Series(np.zeros(12), index=make_index(['date', None, None]))
        store.append('ser', ser)
        xp = Series(np.zeros(12), index=make_index(['date', 'level_1', 'level_2']))
        tm.assert_series_equal(store.select('ser'), xp)
        _maybe_remove(store, 'df')
        df = DataFrame(np.zeros((12, 2)), columns=['a', 'b'], index=make_index(['date', 'a', 't']))
        msg = 'duplicate names/columns in the multi-index when storing as a table'
        with pytest.raises(ValueError, match=msg):
            store.append('df', df)
        _maybe_remove(store, 'df')
        df = DataFrame(np.zeros((12, 2)), columns=['a', 'b'], index=make_index(['date', 'date', 'date']))
        with pytest.raises(ValueError, match=msg):
            store.append('df', df)
        _maybe_remove(store, 'df')
        df = DataFrame(np.zeros((12, 2)), columns=['a', 'b'], index=make_index(['date', 's', 't']))
        store.append('df', df)
        tm.assert_frame_equal(store.select('df'), df)