from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
class TestLocListlike:

    @pytest.mark.parametrize('box', [lambda x: x, np.asarray, list])
    def test_loc_getitem_list_of_labels_categoricalindex_with_na(self, box):
        ci = CategoricalIndex(['A', 'B', np.nan])
        ser = Series(range(3), index=ci)
        result = ser.loc[box(ci)]
        tm.assert_series_equal(result, ser)
        result = ser[box(ci)]
        tm.assert_series_equal(result, ser)
        result = ser.to_frame().loc[box(ci)]
        tm.assert_frame_equal(result, ser.to_frame())
        ser2 = ser[:-1]
        ci2 = ci[1:]
        msg = 'not in index'
        with pytest.raises(KeyError, match=msg):
            ser2.loc[box(ci2)]
        with pytest.raises(KeyError, match=msg):
            ser2[box(ci2)]
        with pytest.raises(KeyError, match=msg):
            ser2.to_frame().loc[box(ci2)]

    def test_loc_getitem_series_label_list_missing_values(self):
        key = np.array(['2001-01-04', '2001-01-02', '2001-01-04', '2001-01-14'], dtype='datetime64')
        ser = Series([2, 5, 8, 11], date_range('2001-01-01', freq='D', periods=4))
        with pytest.raises(KeyError, match='not in index'):
            ser.loc[key]

    def test_loc_getitem_series_label_list_missing_integer_values(self):
        ser = Series(index=np.array([9730701000001104, 10049011000001109]), data=np.array([999000011000001104, 999000011000001104]))
        with pytest.raises(KeyError, match='not in index'):
            ser.loc[np.array([9730701000001104, 10047311000001102])]

    @pytest.mark.parametrize('to_period', [True, False])
    def test_loc_getitem_listlike_of_datetimelike_keys(self, to_period):
        idx = date_range('2011-01-01', '2011-01-02', freq='D', name='idx')
        if to_period:
            idx = idx.to_period('D')
        ser = Series([0.1, 0.2], index=idx, name='s')
        keys = [Timestamp('2011-01-01'), Timestamp('2011-01-02')]
        if to_period:
            keys = [x.to_period('D') for x in keys]
        result = ser.loc[keys]
        exp = Series([0.1, 0.2], index=idx, name='s')
        if not to_period:
            exp.index = exp.index._with_freq(None)
        tm.assert_series_equal(result, exp, check_index_type=True)
        keys = [Timestamp('2011-01-02'), Timestamp('2011-01-02'), Timestamp('2011-01-01')]
        if to_period:
            keys = [x.to_period('D') for x in keys]
        exp = Series([0.2, 0.2, 0.1], index=Index(keys, name='idx', dtype=idx.dtype), name='s')
        result = ser.loc[keys]
        tm.assert_series_equal(result, exp, check_index_type=True)
        keys = [Timestamp('2011-01-03'), Timestamp('2011-01-02'), Timestamp('2011-01-03')]
        if to_period:
            keys = [x.to_period('D') for x in keys]
        with pytest.raises(KeyError, match='not in index'):
            ser.loc[keys]

    def test_loc_named_index(self):
        df = DataFrame([[1, 2], [4, 5], [7, 8]], index=['cobra', 'viper', 'sidewinder'], columns=['max_speed', 'shield'])
        expected = df.iloc[:2]
        expected.index.name = 'foo'
        result = df.loc[Index(['cobra', 'viper'], name='foo')]
        tm.assert_frame_equal(result, expected)