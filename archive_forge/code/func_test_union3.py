from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
@pytest.mark.parametrize('box', [np.array, Series, list])
def test_union3(self, sort, box):
    everything = date_range('2020-01-01', periods=10)
    first = everything[:5]
    second = everything[5:]
    expected = first.union(second, sort=sort)
    case = box(second.values)
    result = first.union(case, sort=sort)
    tm.assert_index_equal(result, expected)