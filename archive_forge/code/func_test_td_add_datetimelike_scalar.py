from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.parametrize('op', [operator.add, ops.radd])
def test_td_add_datetimelike_scalar(self, op):
    td = Timedelta(10, unit='d')
    result = op(td, datetime(2016, 1, 1))
    if op is operator.add:
        assert isinstance(result, Timestamp)
    assert result == Timestamp(2016, 1, 11)
    result = op(td, Timestamp('2018-01-12 18:09'))
    assert isinstance(result, Timestamp)
    assert result == Timestamp('2018-01-22 18:09')
    result = op(td, np.datetime64('2018-01-12'))
    assert isinstance(result, Timestamp)
    assert result == Timestamp('2018-01-22')
    result = op(td, NaT)
    assert result is NaT