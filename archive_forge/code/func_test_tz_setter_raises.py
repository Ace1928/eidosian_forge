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
def test_tz_setter_raises(self):
    arr = DatetimeArray._from_sequence(['2000'], dtype=DatetimeTZDtype(tz='US/Central'))
    with pytest.raises(AttributeError, match='tz_localize'):
        arr.tz = 'UTC'