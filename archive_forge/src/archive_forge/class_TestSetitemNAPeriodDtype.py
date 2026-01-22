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
class TestSetitemNAPeriodDtype(SetitemCastingEquivalents):

    @pytest.fixture
    def expected(self, key):
        exp = Series(period_range('2000-01-01', periods=10, freq='D'))
        exp._values.view('i8')[key] = NaT._value
        assert exp[key] is NaT or all((x is NaT for x in exp[key]))
        return exp

    @pytest.fixture
    def obj(self):
        return Series(period_range('2000-01-01', periods=10, freq='D'))

    @pytest.fixture(params=[3, slice(3, 5)])
    def key(self, request):
        return request.param

    @pytest.fixture(params=[None, np.nan])
    def val(self, request):
        return request.param

    @pytest.fixture
    def warn(self):
        return None