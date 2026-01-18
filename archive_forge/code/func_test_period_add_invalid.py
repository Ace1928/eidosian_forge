from datetime import timedelta
import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas import (
def test_period_add_invalid(self):
    per1 = Period(freq='D', year=2008, month=1, day=1)
    per2 = Period(freq='D', year=2008, month=1, day=2)
    msg = '|'.join(['unsupported operand type\\(s\\)', 'can only concatenate str', 'must be str, not Period'])
    with pytest.raises(TypeError, match=msg):
        per1 + 'str'
    with pytest.raises(TypeError, match=msg):
        'str' + per1
    with pytest.raises(TypeError, match=msg):
        per1 + per2