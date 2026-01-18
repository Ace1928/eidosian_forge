import re
import pytest
from pandas._libs.tslibs import (
@pytest.mark.parametrize('freq_input,expected', [(to_offset('10us'), offsets.Micro(10)), (offsets.Hour(), offsets.Hour()), ('2h30min', offsets.Minute(150)), ('2h 30min', offsets.Minute(150)), ('2h30min15s', offsets.Second(150 * 60 + 15)), ('2h 60min', offsets.Hour(3)), ('2h 20.5min', offsets.Second(8430)), ('1.5min', offsets.Second(90)), ('0.5s', offsets.Milli(500)), ('15ms500us', offsets.Micro(15500)), ('10s75ms', offsets.Milli(10075)), ('1s0.25ms', offsets.Micro(1000250)), ('1s0.25ms', offsets.Micro(1000250)), ('2800ns', offsets.Nano(2800)), ('2SME', offsets.SemiMonthEnd(2)), ('2SME-16', offsets.SemiMonthEnd(2, day_of_month=16)), ('2SMS-14', offsets.SemiMonthBegin(2, day_of_month=14)), ('2SMS-15', offsets.SemiMonthBegin(2))])
def test_to_offset(freq_input, expected):
    result = to_offset(freq_input)
    assert result == expected