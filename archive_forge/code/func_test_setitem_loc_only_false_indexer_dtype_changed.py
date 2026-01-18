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
@pytest.mark.parametrize('box', [list, np.array, Series])
def test_setitem_loc_only_false_indexer_dtype_changed(self, box):
    df = DataFrame({'a': ['a'], 'b': [1], 'c': [1]})
    indexer = box([False])
    df.loc[indexer, ['b']] = 10 - df['c']
    expected = DataFrame({'a': ['a'], 'b': [1], 'c': [1]})
    tm.assert_frame_equal(df, expected)
    df.loc[indexer, ['b']] = 9
    tm.assert_frame_equal(df, expected)