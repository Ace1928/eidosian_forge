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
def test_setitem_slice_float_raises(self, datetime_series):
    msg = 'cannot do slice indexing on DatetimeIndex with these indexers \\[{key}\\] of type float'
    with pytest.raises(TypeError, match=msg.format(key='4\\.0')):
        datetime_series[4.0:10.0] = 0
    with pytest.raises(TypeError, match=msg.format(key='4\\.5')):
        datetime_series[4.5:10.0] = 0