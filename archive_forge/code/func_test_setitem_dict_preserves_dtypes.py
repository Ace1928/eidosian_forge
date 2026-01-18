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
def test_setitem_dict_preserves_dtypes(self):
    expected = DataFrame({'a': Series([0, 1, 2], dtype='int64'), 'b': Series([1, 2, 3], dtype=float), 'c': Series([1, 2, 3], dtype=float), 'd': Series([1, 2, 3], dtype='uint32')})
    df = DataFrame({'a': Series([], dtype='int64'), 'b': Series([], dtype=float), 'c': Series([], dtype=float), 'd': Series([], dtype='uint32')})
    for idx, b in enumerate([1, 2, 3]):
        df.loc[df.shape[0]] = {'a': int(idx), 'b': float(b), 'c': float(b), 'd': np.uint32(b)}
    tm.assert_frame_equal(df, expected)