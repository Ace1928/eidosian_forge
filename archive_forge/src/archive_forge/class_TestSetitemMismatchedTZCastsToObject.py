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
class TestSetitemMismatchedTZCastsToObject(SetitemCastingEquivalents):

    @pytest.fixture
    def obj(self):
        return Series(date_range('2000', periods=2, tz='US/Central'))

    @pytest.fixture
    def val(self):
        return Timestamp('2000', tz='US/Eastern')

    @pytest.fixture
    def key(self):
        return 0

    @pytest.fixture
    def expected(self, obj, val):
        expected = Series([val.tz_convert('US/Central'), Timestamp('2000-01-02 00:00:00-06:00', tz='US/Central')], dtype=obj.dtype)
        return expected

    @pytest.fixture
    def warn(self):
        return None