from datetime import timedelta
import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas import (
def test_period_add_timestamp_raises(self):
    ts = Timestamp('2017')
    per = Period('2017', freq='M')
    msg = "unsupported operand type\\(s\\) for \\+: 'Timestamp' and 'Period'"
    with pytest.raises(TypeError, match=msg):
        ts + per
    msg = "unsupported operand type\\(s\\) for \\+: 'Period' and 'Timestamp'"
    with pytest.raises(TypeError, match=msg):
        per + ts