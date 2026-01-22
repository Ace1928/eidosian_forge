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
class TestSetitemCasting:

    @pytest.mark.parametrize('unique', [True, False])
    @pytest.mark.parametrize('val', [3, 3.0, '3'], ids=type)
    def test_setitem_non_bool_into_bool(self, val, indexer_sli, unique):
        ser = Series([True, False])
        if not unique:
            ser.index = [1, 1]
        with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
            indexer_sli(ser)[1] = val
        assert type(ser.iloc[1]) == type(val)
        expected = Series([True, val], dtype=object, index=ser.index)
        if not unique and indexer_sli is not tm.iloc:
            expected = Series([val, val], dtype=object, index=[1, 1])
        tm.assert_series_equal(ser, expected)

    def test_setitem_boolean_array_into_npbool(self):
        ser = Series([True, False, True])
        values = ser._values
        arr = array([True, False, None])
        ser[:2] = arr[:2]
        assert ser._values is values
        with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
            ser[1:] = arr[1:]
        expected = Series(arr)
        tm.assert_series_equal(ser, expected)