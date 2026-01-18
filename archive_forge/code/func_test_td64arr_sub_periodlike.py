from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('pi_freq', ['D', 'W', 'Q', 'h'])
@pytest.mark.parametrize('tdi_freq', [None, 'h'])
def test_td64arr_sub_periodlike(self, box_with_array, box_with_array2, tdi_freq, pi_freq):
    tdi = TimedeltaIndex(['1 hours', '2 hours'], freq=tdi_freq)
    dti = Timestamp('2018-03-07 17:16:40') + tdi
    pi = dti.to_period(pi_freq)
    per = pi[0]
    tdi = tm.box_expected(tdi, box_with_array)
    pi = tm.box_expected(pi, box_with_array2)
    msg = 'cannot subtract|unsupported operand type'
    with pytest.raises(TypeError, match=msg):
        tdi - pi
    with pytest.raises(TypeError, match=msg):
        tdi - per