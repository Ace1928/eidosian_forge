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
@pytest.mark.parametrize('obj,expected,key,warn', [pytest.param(Series(interval_range(1, 5)), Series([Interval(1, 2), np.nan, Interval(3, 4), Interval(4, 5)], dtype='interval[float64]'), 1, FutureWarning, id='interval_int_na_value'), pytest.param(Series([2, 3, 4, 5, 6, 7, 8, 9, 10]), Series([np.nan, 3, np.nan, 5, np.nan, 7, np.nan, 9, np.nan]), slice(None, None, 2), None, id='int_series_slice_key_step'), pytest.param(Series([True, True, False, False]), Series([np.nan, True, np.nan, False], dtype=object), slice(None, None, 2), FutureWarning, id='bool_series_slice_key_step'), pytest.param(Series(np.arange(10)), Series([np.nan, np.nan, np.nan, np.nan, np.nan, 5, 6, 7, 8, 9]), slice(None, 5), None, id='int_series_slice_key'), pytest.param(Series([1, 2, 3]), Series([np.nan, 2, 3]), 0, None, id='int_series_int_key'), pytest.param(Series([False]), Series([np.nan], dtype=object), 0, FutureWarning, id='bool_series_int_key_change_all'), pytest.param(Series([False, True]), Series([np.nan, True], dtype=object), 0, FutureWarning, id='bool_series_int_key')])
class TestSetitemCastingEquivalents(SetitemCastingEquivalents):

    @pytest.fixture(params=[np.nan, np.float64('NaN'), None, NA])
    def val(self, request):
        """
        NA values that should generally be valid_na for *all* dtypes.

        Include both python float NaN and np.float64; only np.float64 has a
        `dtype` attribute.
        """
        return request.param