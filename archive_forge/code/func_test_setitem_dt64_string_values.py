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
@pytest.mark.parametrize('box', [list, np.array, pd.array, pd.Categorical, Index])
@pytest.mark.parametrize('key', [[0, 1], slice(0, 2), np.array([True, True, False])])
def test_setitem_dt64_string_values(self, tz_naive_fixture, indexer_sli, key, box):
    tz = tz_naive_fixture
    if isinstance(key, slice) and indexer_sli is tm.loc:
        key = slice(0, 1)
    dti = date_range('2016-01-01', periods=3, tz=tz)
    ser = Series(dti.copy(deep=True))
    values = ser._values
    newvals = box(['2019-01-01', '2010-01-02'])
    values._validate_setitem_value(newvals)
    indexer_sli(ser)[key] = newvals
    if tz is None:
        assert ser.dtype == dti.dtype
        assert ser._values._ndarray is values._ndarray
    else:
        assert ser._values is values