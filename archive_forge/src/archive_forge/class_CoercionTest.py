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
class CoercionTest(SetitemCastingEquivalents):

    @pytest.fixture
    def key(self):
        return 1

    @pytest.fixture
    def expected(self, obj, key, val, exp_dtype):
        vals = list(obj)
        vals[key] = val
        return Series(vals, dtype=exp_dtype)