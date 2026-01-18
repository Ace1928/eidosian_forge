import re
import pytest
from pandas._libs.tslibs import (
@pytest.mark.parametrize('freqstr,expected', [('2D 3h', offsets.Hour(51)), ('2 D3 h', offsets.Hour(51)), ('2 D 3 h', offsets.Hour(51)), ('  2 D 3 h  ', offsets.Hour(51)), ('   h    ', offsets.Hour()), (' 3  h    ', offsets.Hour(3))])
def test_to_offset_whitespace(freqstr, expected):
    result = to_offset(freqstr)
    assert result == expected