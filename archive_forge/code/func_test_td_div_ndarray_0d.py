from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_div_ndarray_0d(self):
    td = Timedelta('1 day')
    other = np.array(1)
    res = td / other
    assert isinstance(res, Timedelta)
    assert res == td