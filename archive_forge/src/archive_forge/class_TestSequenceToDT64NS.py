import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
class TestSequenceToDT64NS:

    def test_tz_dtype_mismatch_raises(self):
        arr = DatetimeArray._from_sequence(['2000'], dtype=DatetimeTZDtype(tz='US/Central'))
        with pytest.raises(TypeError, match='data is already tz-aware'):
            DatetimeArray._from_sequence(arr, dtype=DatetimeTZDtype(tz='UTC'))

    def test_tz_dtype_matches(self):
        dtype = DatetimeTZDtype(tz='US/Central')
        arr = DatetimeArray._from_sequence(['2000'], dtype=dtype)
        result = DatetimeArray._from_sequence(arr, dtype=dtype)
        tm.assert_equal(arr, result)

    @pytest.mark.parametrize('order', ['F', 'C'])
    def test_2d(self, order):
        dti = pd.date_range('2016-01-01', periods=6, tz='US/Pacific')
        arr = np.array(dti, dtype=object).reshape(3, 2)
        if order == 'F':
            arr = arr.T
        res = DatetimeArray._from_sequence(arr, dtype=dti.dtype)
        expected = DatetimeArray._from_sequence(arr.ravel(), dtype=dti.dtype).reshape(arr.shape)
        tm.assert_datetime_array_equal(res, expected)