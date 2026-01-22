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
class TestSetitemRangeIntoIntegerSeries(SetitemCastingEquivalents):

    @pytest.fixture
    def obj(self, any_int_numpy_dtype):
        dtype = np.dtype(any_int_numpy_dtype)
        ser = Series(range(5), dtype=dtype)
        return ser

    @pytest.fixture
    def val(self):
        return range(2, 4)

    @pytest.fixture
    def key(self):
        return slice(0, 2)

    @pytest.fixture
    def expected(self, any_int_numpy_dtype):
        dtype = np.dtype(any_int_numpy_dtype)
        exp = Series([2, 3, 2, 3, 4], dtype=dtype)
        return exp

    @pytest.fixture
    def warn(self):
        return None