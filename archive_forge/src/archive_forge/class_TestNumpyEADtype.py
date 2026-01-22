from __future__ import annotations
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.astype import astype_array
import pandas.core.dtypes.common as com
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
import pandas as pd
import pandas._testing as tm
from pandas.api.types import pandas_dtype
from pandas.arrays import SparseArray
class TestNumpyEADtype:

    @pytest.mark.parametrize('box', [pd.Timestamp, 'pd.Timestamp', list])
    def test_invalid_dtype_error(self, box):
        with pytest.raises(TypeError, match='not understood'):
            com.pandas_dtype(box)

    @pytest.mark.parametrize('dtype', [object, 'float64', np.object_, np.dtype('object'), 'O', np.float64, float, np.dtype('float64'), 'object_'])
    def test_pandas_dtype_valid(self, dtype):
        assert com.pandas_dtype(dtype) == dtype

    @pytest.mark.parametrize('dtype', ['M8[ns]', 'm8[ns]', 'object', 'float64', 'int64'])
    def test_numpy_dtype(self, dtype):
        assert com.pandas_dtype(dtype) == np.dtype(dtype)

    def test_numpy_string_dtype(self):
        assert com.pandas_dtype('U') == np.dtype('U')
        assert com.pandas_dtype('S') == np.dtype('S')

    @pytest.mark.parametrize('dtype', ['datetime64[ns, US/Eastern]', 'datetime64[ns, Asia/Tokyo]', 'datetime64[ns, UTC]', 'M8[ns, US/Eastern]', 'M8[ns, Asia/Tokyo]', 'M8[ns, UTC]'])
    def test_datetimetz_dtype(self, dtype):
        assert com.pandas_dtype(dtype) == DatetimeTZDtype.construct_from_string(dtype)
        assert com.pandas_dtype(dtype) == dtype

    def test_categorical_dtype(self):
        assert com.pandas_dtype('category') == CategoricalDtype()

    @pytest.mark.parametrize('dtype', ['period[D]', 'period[3M]', 'period[us]', 'Period[D]', 'Period[3M]', 'Period[us]'])
    def test_period_dtype(self, dtype):
        assert com.pandas_dtype(dtype) is not PeriodDtype(dtype)
        assert com.pandas_dtype(dtype) == PeriodDtype(dtype)
        assert com.pandas_dtype(dtype) == dtype