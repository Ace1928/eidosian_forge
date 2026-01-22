from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import IndexingError
from pandas.core.dtypes.common import is_list_like
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
class TestSetitemScalarIndexer:

    def test_setitem_negative_out_of_bounds(self):
        ser = Series(['a'] * 10, index=['a'] * 10)
        msg = 'index -11|-1 is out of bounds for axis 0 with size 10'
        warn_msg = 'Series.__setitem__ treating keys as positions is deprecated'
        with pytest.raises(IndexError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=warn_msg):
                ser[-11] = 'foo'

    @pytest.mark.parametrize('indexer', [tm.loc, tm.at])
    @pytest.mark.parametrize('ser_index', [0, 1])
    def test_setitem_series_object_dtype(self, indexer, ser_index):
        ser = Series([0, 0], dtype='object')
        idxr = indexer(ser)
        idxr[0] = Series([42], index=[ser_index])
        expected = Series([Series([42], index=[ser_index]), 0], dtype='object')
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize('index, exp_value', [(0, 42), (1, np.nan)])
    def test_setitem_series(self, index, exp_value):
        ser = Series([0, 0])
        ser.loc[0] = Series([42], index=[index])
        expected = Series([exp_value, 0])
        tm.assert_series_equal(ser, expected)