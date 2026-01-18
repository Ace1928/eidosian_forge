from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_rfloordiv_numeric_scalar(self):
    td = Timedelta(hours=3, minutes=3)
    assert td.__rfloordiv__(np.nan) is NotImplemented
    assert td.__rfloordiv__(3.5) is NotImplemented
    assert td.__rfloordiv__(2) is NotImplemented
    assert td.__rfloordiv__(np.float64(2.0)) is NotImplemented
    assert td.__rfloordiv__(np.uint8(9)) is NotImplemented
    assert td.__rfloordiv__(np.int32(2.0)) is NotImplemented
    msg = "unsupported operand type\\(s\\) for //: '.*' and 'Timedelta"
    with pytest.raises(TypeError, match=msg):
        np.float64(2.0) // td
    with pytest.raises(TypeError, match=msg):
        np.uint8(9) // td
    with pytest.raises(TypeError, match=msg):
        np.int32(2.0) // td