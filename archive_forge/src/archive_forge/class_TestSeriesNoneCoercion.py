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
@pytest.mark.parametrize('obj,expected,warn', [(Series([1, 2, 3]), Series([np.nan, 2, 3]), None), (Series([1.0, 2.0, 3.0]), Series([np.nan, 2.0, 3.0]), None), (Series([datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)]), Series([NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)]), None), (Series(['foo', 'bar', 'baz']), Series([None, 'bar', 'baz']), None)])
class TestSeriesNoneCoercion(SetitemCastingEquivalents):

    @pytest.fixture
    def key(self):
        return 0

    @pytest.fixture
    def val(self):
        return None