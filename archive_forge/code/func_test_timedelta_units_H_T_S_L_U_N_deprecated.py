import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
@pytest.mark.parametrize('depr_unit, unit', [('H', 'hour'), ('T', 'minute'), ('t', 'minute'), ('S', 'second'), ('L', 'millisecond'), ('l', 'millisecond'), ('U', 'microsecond'), ('u', 'microsecond'), ('N', 'nanosecond'), ('n', 'nanosecond')])
def test_timedelta_units_H_T_S_L_U_N_deprecated(self, depr_unit, unit):
    depr_msg = f"'{depr_unit}' is deprecated and will be removed in a future version."
    expected = to_timedelta(np.arange(5), unit=unit)
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        result = to_timedelta(np.arange(5), unit=depr_unit)
        tm.assert_index_equal(result, expected)