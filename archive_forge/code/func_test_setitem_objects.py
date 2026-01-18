from __future__ import annotations
from datetime import timedelta
import operator
import numpy as np
import pytest
from pandas._libs.tslibs import tz_compare
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('obj', [pd.Timestamp('2021-01-01'), pd.Timestamp('2021-01-01').to_datetime64(), pd.Timestamp('2021-01-01').to_pydatetime()])
def test_setitem_objects(self, obj):
    dti = pd.date_range('2000', periods=2, freq='D')
    arr = dti._data
    arr[0] = obj
    assert arr[0] == obj