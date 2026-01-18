from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
def test_value_counts_series(self):
    values = np.array([3, 1, 2, 3, 4, np.nan])
    result = Series(values).value_counts(bins=3)
    expected = Series([2, 2, 1], index=IntervalIndex.from_tuples([(0.996, 2.0), (2.0, 3.0), (3.0, 4.0)], dtype='interval[float64, right]'), name='count')
    tm.assert_series_equal(result, expected)