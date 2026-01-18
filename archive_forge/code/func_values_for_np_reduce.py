from collections import deque
import re
import string
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.arrays import SparseArray
@pytest.fixture(params=[pd.array([1, 3, 2], dtype=np.int64), pd.array([1, 3, 2], dtype='Int64'), pd.array([1, 3, 2], dtype='Float32'), pd.array([1, 10, 2], dtype='Sparse[int]'), pd.to_datetime(['2000', '2010', '2001']), pd.to_datetime(['2000', '2010', '2001']).tz_localize('CET'), pd.to_datetime(['2000', '2010', '2001']).to_period(freq='D'), pd.to_timedelta(['1 Day', '3 Days', '2 Days']), pd.IntervalIndex([pd.Interval(0, 1), pd.Interval(2, 3), pd.Interval(1, 2)])], ids=lambda x: str(x.dtype))
def values_for_np_reduce(request):
    return request.param