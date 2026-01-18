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
def test_setitem_multi_index(self):
    it = (['jim', 'joe', 'jolie'], ['first', 'last'], ['left', 'center', 'right'])
    cols = MultiIndex.from_product(it)
    index = date_range('20141006', periods=20)
    vals = np.random.default_rng(2).integers(1, 1000, (len(index), len(cols)))
    df = DataFrame(vals, columns=cols, index=index)
    i, j = (df.index.values.copy(), it[-1][:])
    np.random.default_rng(2).shuffle(i)
    df['jim'] = df['jolie'].loc[i, ::-1]
    tm.assert_frame_equal(df['jim'], df['jolie'])
    np.random.default_rng(2).shuffle(j)
    df['joe', 'first'] = df['jolie', 'last'].loc[i, j]
    tm.assert_frame_equal(df['joe', 'first'], df['jolie', 'last'])
    np.random.default_rng(2).shuffle(j)
    df['joe', 'last'] = df['jolie', 'first'].loc[i, j]
    tm.assert_frame_equal(df['joe', 'last'], df['jolie', 'first'])