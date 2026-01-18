from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
def test_sub_dti_dti(self, unit):
    dti = date_range('20130101', periods=3, unit=unit)
    dti_tz = date_range('20130101', periods=3, unit=unit).tz_localize('US/Eastern')
    expected = TimedeltaIndex([0, 0, 0]).as_unit(unit)
    result = dti - dti
    tm.assert_index_equal(result, expected)
    result = dti_tz - dti_tz
    tm.assert_index_equal(result, expected)
    msg = 'Cannot subtract tz-naive and tz-aware datetime-like objects'
    with pytest.raises(TypeError, match=msg):
        dti_tz - dti
    with pytest.raises(TypeError, match=msg):
        dti - dti_tz
    dti -= dti
    tm.assert_index_equal(dti, expected)
    dti1 = date_range('20130101', periods=3, unit=unit)
    dti2 = date_range('20130101', periods=4, unit=unit)
    msg = 'cannot add indices of unequal length'
    with pytest.raises(ValueError, match=msg):
        dti1 - dti2
    dti1 = DatetimeIndex(['2012-01-01', np.nan, '2012-01-03']).as_unit(unit)
    dti2 = DatetimeIndex(['2012-01-02', '2012-01-03', np.nan]).as_unit(unit)
    expected = TimedeltaIndex(['1 days', np.nan, np.nan]).as_unit(unit)
    result = dti2 - dti1
    tm.assert_index_equal(result, expected)