from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_add_sub_numeric_raises(self):
    td = Timedelta(10, unit='d')
    msg = 'unsupported operand type'
    for other in [2, 2.0, np.int64(2), np.float64(2)]:
        with pytest.raises(TypeError, match=msg):
            td + other
        with pytest.raises(TypeError, match=msg):
            other + td
        with pytest.raises(TypeError, match=msg):
            td - other
        with pytest.raises(TypeError, match=msg):
            other - td