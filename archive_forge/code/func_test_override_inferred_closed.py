from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_unsigned_integer_dtype
from pandas.core.dtypes.dtypes import IntervalDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
import pandas.core.common as com
@pytest.mark.parametrize('data, closed', [([], 'both'), ([np.nan, np.nan], 'neither'), ([Interval(0, 3, closed='neither'), Interval(2, 5, closed='neither')], 'left'), ([Interval(0, 3, closed='left'), Interval(2, 5, closed='right')], 'neither'), (IntervalIndex.from_breaks(range(5), closed='both'), 'right')])
def test_override_inferred_closed(self, constructor, data, closed):
    if isinstance(data, IntervalIndex):
        tuples = data.to_tuples()
    else:
        tuples = [(iv.left, iv.right) if notna(iv) else iv for iv in data]
    expected = IntervalIndex.from_tuples(tuples, closed=closed)
    result = constructor(data, closed=closed)
    tm.assert_index_equal(result, expected)