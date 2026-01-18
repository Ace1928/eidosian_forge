from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
def test_prod_sum_min_count_mixed_object():
    df = DataFrame([1, 'a', True])
    result = df.prod(axis=0, min_count=1, numeric_only=False)
    expected = Series(['a'], dtype=object)
    tm.assert_series_equal(result, expected)
    msg = re.escape("unsupported operand type(s) for +: 'int' and 'str'")
    with pytest.raises(TypeError, match=msg):
        df.sum(axis=0, min_count=1, numeric_only=False)