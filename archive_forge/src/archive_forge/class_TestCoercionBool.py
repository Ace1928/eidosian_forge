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
@pytest.mark.parametrize('val,exp_dtype,warn', [(1, object, FutureWarning), ('3', object, FutureWarning), (3, object, FutureWarning), (1.1, object, FutureWarning), (1 + 1j, object, FutureWarning), (True, bool, None)])
class TestCoercionBool(CoercionTest):

    @pytest.fixture
    def obj(self):
        return Series([True, False, True, False], dtype=bool)