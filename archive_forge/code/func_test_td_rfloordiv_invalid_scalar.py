from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_rfloordiv_invalid_scalar(self):
    td = Timedelta(hours=3, minutes=3)
    dt64 = np.datetime64('2016-01-01', 'us')
    assert td.__rfloordiv__(dt64) is NotImplemented
    msg = "unsupported operand type\\(s\\) for //: 'numpy.datetime64' and 'Timedelta'"
    with pytest.raises(TypeError, match=msg):
        dt64 // td