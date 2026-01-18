import re
import pytest
from pandas._libs.tslibs import (
@pytest.mark.filterwarnings("ignore:.*'m' is deprecated.*:FutureWarning")
@pytest.mark.parametrize('freqstr', ['2h20m', 'us1', '-us', '3us1', '-2-3us', '-2D:3h', '1.5.0s', '2SMS-15-15', '2SMS-15D', '100foo', '+-1d', '-+1h', '+1', '-7', '+d', '-m', 'SME-0', 'SME-28', 'SME-29', 'SME-FOO', 'BSM', 'SME--1', 'SMS-1', 'SMS-28', 'SMS-30', 'SMS-BAR', 'SMS-BYR', 'BSMS', 'SMS--2'])
def test_to_offset_invalid(freqstr):
    msg = re.escape(f'Invalid frequency: {freqstr}')
    with pytest.raises(ValueError, match=msg):
        to_offset(freqstr)