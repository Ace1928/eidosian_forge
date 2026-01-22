import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestPeriodRangeDisallowedFreqs:

    def test_constructor_U(self):
        with pytest.raises(ValueError, match='Invalid frequency: X'):
            period_range('2007-1-1', periods=500, freq='X')

    @pytest.mark.parametrize('freq,freq_depr', [('2Y', '2A'), ('2Y', '2a'), ('2Y-AUG', '2A-AUG'), ('2Y-AUG', '2A-aug')])
    def test_a_deprecated_from_time_series(self, freq, freq_depr):
        msg = f"'{freq_depr[1:]}' is deprecated and will be removed in a "
        f"future version. Please use '{freq[1:]}' instead."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            period_range(freq=freq_depr, start='1/1/2001', end='12/1/2009')

    @pytest.mark.parametrize('freq_depr', ['2H', '2MIN', '2S', '2US', '2NS'])
    def test_uppercase_freq_deprecated_from_time_series(self, freq_depr):
        msg = f"'{freq_depr[1:]}' is deprecated and will be removed in a "
        f"future version. Please use '{freq_depr.lower()[1:]}' instead."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            period_range('2020-01-01 00:00:00 00:00', periods=2, freq=freq_depr)

    @pytest.mark.parametrize('freq_depr', ['2m', '2q-sep', '2y', '2w'])
    def test_lowercase_freq_deprecated_from_time_series(self, freq_depr):
        msg = f"'{freq_depr[1:]}' is deprecated and will be removed in a "
        f"future version. Please use '{freq_depr.upper()[1:]}' instead."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            period_range(freq=freq_depr, start='1/1/2001', end='12/1/2009')