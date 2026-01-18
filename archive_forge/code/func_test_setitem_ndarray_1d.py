import array
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import IndexingError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj
def test_setitem_ndarray_1d(self):
    df = DataFrame(index=Index(np.arange(1, 11), dtype=np.int64))
    df['foo'] = np.zeros(10, dtype=np.float64)
    df['bar'] = np.zeros(10, dtype=complex)
    msg = 'Must have equal len keys and value when setting with an iterable'
    with pytest.raises(ValueError, match=msg):
        df.loc[df.index[2:5], 'bar'] = np.array([2.33j, 1.23 + 0.1j, 2.2, 1.0])
    df.loc[df.index[2:6], 'bar'] = np.array([2.33j, 1.23 + 0.1j, 2.2, 1.0])
    result = df.loc[df.index[2:6], 'bar']
    expected = Series([2.33j, 1.23 + 0.1j, 2.2, 1.0], index=[3, 4, 5, 6], name='bar')
    tm.assert_series_equal(result, expected)