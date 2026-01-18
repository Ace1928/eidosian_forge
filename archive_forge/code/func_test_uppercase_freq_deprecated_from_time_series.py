import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('freq_depr', ['2H', '2MIN', '2S', '2US', '2NS'])
def test_uppercase_freq_deprecated_from_time_series(self, freq_depr):
    msg = f"'{freq_depr[1:]}' is deprecated and will be removed in a "
    f"future version. Please use '{freq_depr.lower()[1:]}' instead."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        period_range('2020-01-01 00:00:00 00:00', periods=2, freq=freq_depr)