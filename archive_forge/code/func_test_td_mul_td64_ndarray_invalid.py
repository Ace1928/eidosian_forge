from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_mul_td64_ndarray_invalid(self):
    td = Timedelta('1 day')
    other = np.array([Timedelta('2 Days').to_timedelta64()])
    msg = f"ufunc '?multiply'? cannot use operands with types dtype\\('{tm.ENDIAN}m8\\[ns\\]'\\) and dtype\\('{tm.ENDIAN}m8\\[ns\\]'\\)"
    with pytest.raises(TypeError, match=msg):
        td * other
    with pytest.raises(TypeError, match=msg):
        other * td