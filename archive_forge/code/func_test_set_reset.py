from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_set_reset(self):
    idx = Index([2 ** 63, 2 ** 63 + 5, 2 ** 63 + 10], name='foo')
    df = DataFrame({'A': [0, 1, 2]}, index=idx)
    result = df.reset_index()
    assert result['foo'].dtype == np.dtype('uint64')
    df = result.set_index('foo')
    tm.assert_index_equal(df.index, idx)