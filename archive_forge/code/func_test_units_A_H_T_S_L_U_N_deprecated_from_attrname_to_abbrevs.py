import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
import pandas._testing as tm
@pytest.mark.parametrize('freq', ['A', 'H', 'T', 'S', 'L', 'U', 'N'])
def test_units_A_H_T_S_L_U_N_deprecated_from_attrname_to_abbrevs(freq):
    msg = f"'{freq}' is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        Resolution.get_reso_from_freqstr(freq)