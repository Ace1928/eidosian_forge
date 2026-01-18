from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_union_freq_infer(self):
    dti = date_range('2016-01-01', periods=5)
    left = dti[[0, 1, 3, 4]]
    right = dti[[2, 3, 1]]
    assert left.freq is None
    assert right.freq is None
    result = left.union(right)
    tm.assert_index_equal(result, dti)
    assert result.freq == 'D'