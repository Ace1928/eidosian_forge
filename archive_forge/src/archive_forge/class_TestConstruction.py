from datetime import datetime
import sys
import numpy as np
import pytest
from pandas.compat import PYPY
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.accessor import PandasDelegate
from pandas.core.base import (
class TestConstruction:

    @pytest.mark.parametrize('a', [np.array(['2263-01-01'], dtype='datetime64[D]'), np.array([datetime(2263, 1, 1)], dtype=object), np.array([np.datetime64('2263-01-01', 'D')], dtype=object), np.array(['2263-01-01'], dtype=object)], ids=['datetime64[D]', 'object-datetime.datetime', 'object-numpy-scalar', 'object-string'])
    def test_constructor_datetime_outofbound(self, a, constructor, request, using_infer_string):
        if a.dtype.kind == 'M':
            result = constructor(a)
            assert result.dtype == 'M8[s]'
        else:
            result = constructor(a)
            if using_infer_string and 'object-string' in request.node.callspec.id:
                assert result.dtype == 'string'
            else:
                assert result.dtype == 'object'
            tm.assert_numpy_array_equal(result.to_numpy(), a)
        msg = 'Out of bounds|Out of bounds .* present at position 0'
        with pytest.raises(pd.errors.OutOfBoundsDatetime, match=msg):
            constructor(a, dtype='datetime64[ns]')

    def test_constructor_datetime_nonns(self, constructor):
        arr = np.array(['2020-01-01T00:00:00.000000'], dtype='datetime64[us]')
        dta = pd.core.arrays.DatetimeArray._simple_new(arr, dtype=arr.dtype)
        expected = constructor(dta)
        assert expected.dtype == arr.dtype
        result = constructor(arr)
        tm.assert_equal(result, expected)
        arr.flags.writeable = False
        result = constructor(arr)
        tm.assert_equal(result, expected)