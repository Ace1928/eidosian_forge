import numpy as np
import pytest
from pandas.compat import PY311
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_set_empty_level():
    midx = MultiIndex.from_arrays([[]], names=['A'])
    result = midx.set_levels(pd.DatetimeIndex([]), level=0)
    expected = MultiIndex.from_arrays([pd.DatetimeIndex([])], names=['A'])
    tm.assert_index_equal(result, expected)