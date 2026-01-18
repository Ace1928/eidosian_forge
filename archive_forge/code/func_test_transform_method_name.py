from itertools import chain
import operator
import numpy as np
import pytest
from pandas.core.dtypes.common import is_number
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import (
@pytest.mark.parametrize('method', ['abs', 'shift', 'pct_change', 'cumsum', 'rank'])
def test_transform_method_name(method):
    df = DataFrame({'A': [-1, 2]})
    result = df.transform(method)
    expected = operator.methodcaller(method)(df)
    tm.assert_frame_equal(result, expected)