from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_div_nat_invalid(self, box_with_array):
    rng = timedelta_range('1 days', '10 days', name='foo')
    rng = tm.box_expected(rng, box_with_array)
    with pytest.raises(TypeError, match='unsupported operand type'):
        rng / NaT
    with pytest.raises(TypeError, match='Cannot divide NaTType by'):
        NaT / rng
    dt64nat = np.datetime64('NaT', 'ns')
    msg = '|'.join(["ufunc '(true_divide|divide)' cannot use operands", 'cannot perform __r?truediv__', 'Cannot divide datetime64 by TimedeltaArray'])
    with pytest.raises(TypeError, match=msg):
        rng / dt64nat
    with pytest.raises(TypeError, match=msg):
        dt64nat / rng