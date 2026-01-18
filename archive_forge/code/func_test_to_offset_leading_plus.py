import re
import pytest
from pandas._libs.tslibs import (
@pytest.mark.parametrize('freqstr,expected', [('+1d', 1), ('+2h30min', 150)])
def test_to_offset_leading_plus(freqstr, expected):
    result = to_offset(freqstr)
    assert result.n == expected