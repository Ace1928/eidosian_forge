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
@pytest.mark.parametrize('val', [2 ** 33 + 1.0, 2 ** 33 + 1.1, 2 ** 62])
class TestSmallIntegerSetitemUpcast(SetitemCastingEquivalents):

    @pytest.fixture
    def obj(self):
        return Series([1, 2, 3], dtype='i4')

    @pytest.fixture
    def key(self):
        return 0

    @pytest.fixture
    def expected(self, val):
        if val % 1 != 0:
            dtype = 'f8'
        else:
            dtype = 'i8'
        return Series([val, 2, 3], dtype=dtype)

    @pytest.fixture
    def warn(self):
        return FutureWarning