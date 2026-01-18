import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_series_groupy_agg_list_like_func_with_args():
    s = Series([1, 2, 3])
    sgb = s.groupby(s)

    def foo1(x, a=1, c=0):
        return x.sum() + a + c

    def foo2(x, b=2, c=0):
        return x.sum() + b + c
    msg = "foo1\\(\\) got an unexpected keyword argument 'b'"
    with pytest.raises(TypeError, match=msg):
        sgb.agg([foo1, foo2], 3, b=3, c=4)
    result = sgb.agg([foo1, foo2], 3, c=4)
    expected = DataFrame([[8, 8], [9, 9], [10, 10]], index=Index([1, 2, 3]), columns=['foo1', 'foo2'])
    tm.assert_frame_equal(result, expected)