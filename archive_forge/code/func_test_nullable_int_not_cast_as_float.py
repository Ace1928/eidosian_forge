import builtins
from io import StringIO
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.tests.groupby import get_groupby_method_args
from pandas.util import _test_decorators as td
@td.skip_if_32bit
@pytest.mark.parametrize('method', ['cummin', 'cummax'])
@pytest.mark.parametrize('dtype,val', [('UInt64', np.iinfo('uint64').max), ('Int64', 2 ** 53 + 1)])
def test_nullable_int_not_cast_as_float(method, dtype, val):
    data = [val, pd.NA]
    df = DataFrame({'grp': [1, 1], 'b': data}, dtype=dtype)
    grouped = df.groupby('grp')
    result = grouped.transform(method)
    expected = DataFrame({'b': data}, dtype=dtype)
    tm.assert_frame_equal(result, expected)