from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
def test_setitem_timestamp_empty_columns(self):
    df = DataFrame(index=range(3))
    df['now'] = Timestamp('20130101', tz='UTC').as_unit('ns')
    expected = DataFrame([[Timestamp('20130101', tz='UTC')]] * 3, index=[0, 1, 2], columns=['now'])
    tm.assert_frame_equal(df, expected)