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
@pytest.mark.parametrize('val,exp_dtype,warn', [(1, np.int64, None), (1.1, np.float64, FutureWarning), (1 + 1j, np.complex128, FutureWarning), (True, object, FutureWarning)])
class TestCoercionInt64(CoercionTest):

    @pytest.fixture
    def obj(self):
        return Series([1, 2, 3, 4])