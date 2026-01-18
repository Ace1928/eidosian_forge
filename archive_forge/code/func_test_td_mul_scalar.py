from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.parametrize('op', [operator.mul, ops.rmul])
def test_td_mul_scalar(self, op):
    td = Timedelta(minutes=3)
    result = op(td, 2)
    assert result == Timedelta(minutes=6)
    result = op(td, 1.5)
    assert result == Timedelta(minutes=4, seconds=30)
    assert op(td, np.nan) is NaT
    assert op(-1, td)._value == -1 * td._value
    assert op(-1.0, td)._value == -1.0 * td._value
    msg = 'unsupported operand type'
    with pytest.raises(TypeError, match=msg):
        op(td, Timestamp(2016, 1, 2))
    with pytest.raises(TypeError, match=msg):
        op(td, td)