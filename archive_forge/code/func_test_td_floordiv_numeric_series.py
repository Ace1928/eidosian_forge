from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_floordiv_numeric_series(self):
    td = Timedelta(hours=3, minutes=4)
    ser = pd.Series([1], dtype=np.int64)
    res = td // ser
    assert res.dtype.kind == 'm'