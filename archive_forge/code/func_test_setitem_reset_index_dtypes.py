from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
def test_setitem_reset_index_dtypes(self):
    df = DataFrame(columns=['a', 'b', 'c']).astype({'a': 'datetime64[ns]', 'b': np.int64, 'c': np.float64})
    df1 = df.set_index(['a'])
    df1['d'] = []
    result = df1.reset_index()
    expected = DataFrame(columns=['a', 'b', 'c', 'd'], index=range(0)).astype({'a': 'datetime64[ns]', 'b': np.int64, 'c': np.float64, 'd': np.float64})
    tm.assert_frame_equal(result, expected)
    df2 = df.set_index(['a', 'b'])
    df2['d'] = []
    result = df2.reset_index()
    tm.assert_frame_equal(result, expected)