from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_transform_doesnt_clobber_ints():
    n = 6
    x = np.arange(n)
    df = DataFrame({'a': x // 2, 'b': 2.0 * x, 'c': 3.0 * x})
    df2 = DataFrame({'a': x // 2 * 1.0, 'b': 2.0 * x, 'c': 3.0 * x})
    gb = df.groupby('a')
    result = gb.transform('mean')
    gb2 = df2.groupby('a')
    expected = gb2.transform('mean')
    tm.assert_frame_equal(result, expected)