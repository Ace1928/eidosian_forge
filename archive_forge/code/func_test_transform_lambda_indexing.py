import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_lambda_indexing():
    df = DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'flux', 'foo', 'flux'], 'B': ['one', 'one', 'two', 'three', 'two', 'six', 'five', 'three'], 'C': range(8), 'D': range(8), 'E': range(8)})
    df = df.set_index(['A', 'B'])
    df = df.sort_index()
    result = df.groupby(level='A').transform(lambda x: x.iloc[-1])
    expected = DataFrame({'C': [3, 3, 7, 7, 4, 4, 4, 4], 'D': [3, 3, 7, 7, 4, 4, 4, 4], 'E': [3, 3, 7, 7, 4, 4, 4, 4]}, index=MultiIndex.from_tuples([('bar', 'one'), ('bar', 'three'), ('flux', 'six'), ('flux', 'three'), ('foo', 'five'), ('foo', 'one'), ('foo', 'two'), ('foo', 'two')], names=['A', 'B']))
    tm.assert_frame_equal(result, expected)