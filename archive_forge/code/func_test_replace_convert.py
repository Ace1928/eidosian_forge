from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't set float into string")
def test_replace_convert(self):
    df = DataFrame([['foo', 'bar', 'bah'], ['bar', 'foo', 'bah']])
    m = {'foo': 1, 'bar': 2, 'bah': 3}
    msg = 'Downcasting behavior in `replace` '
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rep = df.replace(m)
    expec = Series([np.int64] * 3)
    res = rep.dtypes
    tm.assert_series_equal(expec, res)