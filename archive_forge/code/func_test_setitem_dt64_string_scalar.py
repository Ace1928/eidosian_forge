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
def test_setitem_dt64_string_scalar(self, tz_naive_fixture, indexer_sli):
    tz = tz_naive_fixture
    dti = date_range('2016-01-01', periods=3, tz=tz)
    ser = Series(dti.copy(deep=True))
    values = ser._values
    newval = '2018-01-01'
    values._validate_setitem_value(newval)
    indexer_sli(ser)[0] = newval
    if tz is None:
        assert ser.dtype == dti.dtype
        assert ser._values._ndarray is values._ndarray
    else:
        assert ser._values is values