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
def test_setitem_dtypes_bytes_type_to_object(self):
    index = Series(name='id', dtype='S24')
    df = DataFrame(index=index)
    df['a'] = Series(name='a', index=index, dtype=np.uint32)
    df['b'] = Series(name='b', index=index, dtype='S64')
    df['c'] = Series(name='c', index=index, dtype='S64')
    df['d'] = Series(name='d', index=index, dtype=np.uint8)
    result = df.dtypes
    expected = Series([np.uint32, object, object, np.uint8], index=list('abcd'))
    tm.assert_series_equal(result, expected)