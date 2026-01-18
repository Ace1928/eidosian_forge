from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('obox', [np.array, Index, Series])
def test_td64arr_addsub_anchored_offset_arraylike(self, obox, box_with_array):
    tdi = TimedeltaIndex(['1 days 00:00:00', '3 days 04:00:00'])
    tdi = tm.box_expected(tdi, box_with_array)
    anchored = obox([offsets.MonthEnd(), offsets.Day(n=2)])
    msg = 'has incorrect type|cannot add the type MonthEnd'
    with pytest.raises(TypeError, match=msg):
        with tm.assert_produces_warning(PerformanceWarning):
            tdi + anchored
    with pytest.raises(TypeError, match=msg):
        with tm.assert_produces_warning(PerformanceWarning):
            anchored + tdi
    with pytest.raises(TypeError, match=msg):
        with tm.assert_produces_warning(PerformanceWarning):
            tdi - anchored
    with pytest.raises(TypeError, match=msg):
        with tm.assert_produces_warning(PerformanceWarning):
            anchored - tdi