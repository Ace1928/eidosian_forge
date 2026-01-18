from __future__ import annotations
from datetime import timedelta
import operator
import numpy as np
import pytest
from pandas._libs.tslibs import tz_compare
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
def test_shift_value_tzawareness_mismatch(self):
    dti = pd.date_range('2016-01-01', periods=3)
    dta = dti._data
    fv = dta[-1].tz_localize('UTC')
    for invalid in [fv, fv.to_pydatetime()]:
        with pytest.raises(TypeError, match='Cannot compare'):
            dta.shift(1, fill_value=invalid)
    dta = dta.tz_localize('UTC')
    fv = dta[-1].tz_localize(None)
    for invalid in [fv, fv.to_pydatetime(), fv.to_datetime64()]:
        with pytest.raises(TypeError, match='Cannot compare'):
            dta.shift(1, fill_value=invalid)