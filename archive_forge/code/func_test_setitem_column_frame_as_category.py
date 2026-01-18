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
def test_setitem_column_frame_as_category(self):
    df = DataFrame([1, 2, 3])
    df['col1'] = DataFrame([1, 2, 3], dtype='category')
    df['col2'] = Series([1, 2, 3], dtype='category')
    expected_types = Series(['int64', 'category', 'category'], index=[0, 'col1', 'col2'], dtype=object)
    tm.assert_series_equal(df.dtypes, expected_types)