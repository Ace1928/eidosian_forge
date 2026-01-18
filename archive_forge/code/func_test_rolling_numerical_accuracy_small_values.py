from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_numerical_accuracy_small_values():
    s = Series(data=[0.00012456, 0.0003, -0.0, -0.0], index=date_range('1999-02-03', '1999-02-06'))
    result = s.rolling(1).mean()
    tm.assert_series_equal(result, s)