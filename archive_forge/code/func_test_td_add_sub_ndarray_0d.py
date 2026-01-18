from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_add_sub_ndarray_0d(self):
    td = Timedelta('1 day')
    other = np.array(td.asm8)
    result = td + other
    assert isinstance(result, Timedelta)
    assert result == 2 * td
    result = other + td
    assert isinstance(result, Timedelta)
    assert result == 2 * td
    result = other - td
    assert isinstance(result, Timedelta)
    assert result == 0 * td
    result = td - other
    assert isinstance(result, Timedelta)
    assert result == 0 * td