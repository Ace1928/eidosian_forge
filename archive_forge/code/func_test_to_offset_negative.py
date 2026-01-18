import re
import pytest
from pandas._libs.tslibs import (
@pytest.mark.parametrize('freqstr,expected', [('-1s', -1), ('-2SME', -2), ('-1SMS', -1), ('-5min10s', -310)])
def test_to_offset_negative(freqstr, expected):
    result = to_offset(freqstr)
    assert result.n == expected