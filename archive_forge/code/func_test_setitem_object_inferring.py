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
def test_setitem_object_inferring(self):
    idx = Index([Timestamp('2019-12-31')], dtype=object)
    df = DataFrame({'a': [1]})
    with tm.assert_produces_warning(FutureWarning, match='infer'):
        df.loc[:, 'b'] = idx
    with tm.assert_produces_warning(FutureWarning, match='infer'):
        df['c'] = idx
    expected = DataFrame({'a': [1], 'b': Series([Timestamp('2019-12-31')], dtype='datetime64[ns]'), 'c': Series([Timestamp('2019-12-31')], dtype='datetime64[ns]')})
    tm.assert_frame_equal(df, expected)