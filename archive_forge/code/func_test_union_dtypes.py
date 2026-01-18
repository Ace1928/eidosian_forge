from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.parametrize('left, right, expected', [('int64', 'int64', 'int64'), ('int64', 'uint64', 'object'), ('int64', 'float64', 'float64'), ('uint64', 'float64', 'float64'), ('uint64', 'uint64', 'uint64'), ('float64', 'float64', 'float64'), ('datetime64[ns]', 'int64', 'object'), ('datetime64[ns]', 'uint64', 'object'), ('datetime64[ns]', 'float64', 'object'), ('datetime64[ns, CET]', 'int64', 'object'), ('datetime64[ns, CET]', 'uint64', 'object'), ('datetime64[ns, CET]', 'float64', 'object'), ('Period[D]', 'int64', 'object'), ('Period[D]', 'uint64', 'object'), ('Period[D]', 'float64', 'object')])
@pytest.mark.parametrize('names', [('foo', 'foo', 'foo'), ('foo', 'bar', None)])
def test_union_dtypes(left, right, expected, names):
    left = pandas_dtype(left)
    right = pandas_dtype(right)
    a = Index([], dtype=left, name=names[0])
    b = Index([], dtype=right, name=names[1])
    result = a.union(b)
    assert result.dtype == expected
    assert result.name == names[2]
    result = a.intersection(b)
    assert result.name == names[2]