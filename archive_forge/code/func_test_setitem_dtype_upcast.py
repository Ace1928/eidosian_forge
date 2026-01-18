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
def test_setitem_dtype_upcast(self):
    df = DataFrame([{'a': 1}, {'a': 3, 'b': 2}])
    df['c'] = np.nan
    assert df['c'].dtype == np.float64
    with tm.assert_produces_warning(FutureWarning, match='item of incompatible dtype'):
        df.loc[0, 'c'] = 'foo'
    expected = DataFrame({'a': [1, 3], 'b': [np.nan, 2], 'c': Series(['foo', np.nan], dtype=object)})
    tm.assert_frame_equal(df, expected)