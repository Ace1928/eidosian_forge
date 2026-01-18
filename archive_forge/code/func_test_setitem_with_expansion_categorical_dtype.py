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
def test_setitem_with_expansion_categorical_dtype(self):
    df = DataFrame({'value': np.array(np.random.default_rng(2).integers(0, 10000, 100), dtype='int32')})
    labels = Categorical([f'{i} - {i + 499}' for i in range(0, 10000, 500)])
    df = df.sort_values(by=['value'], ascending=True)
    ser = cut(df.value, range(0, 10500, 500), right=False, labels=labels)
    cat = ser.values
    df['D'] = cat
    result = df.dtypes
    expected = Series([np.dtype('int32'), CategoricalDtype(categories=labels, ordered=False)], index=['value', 'D'])
    tm.assert_series_equal(result, expected)
    df['E'] = ser
    result = df.dtypes
    expected = Series([np.dtype('int32'), CategoricalDtype(categories=labels, ordered=False), CategoricalDtype(categories=labels, ordered=False)], index=['value', 'D', 'E'])
    tm.assert_series_equal(result, expected)
    result1 = df['D']
    result2 = df['E']
    tm.assert_categorical_equal(result1._mgr.array, cat)
    ser.name = 'E'
    tm.assert_series_equal(result2.sort_index(), ser.sort_index())