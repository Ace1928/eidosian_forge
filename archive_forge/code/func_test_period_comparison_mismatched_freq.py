from datetime import timedelta
import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas import (
def test_period_comparison_mismatched_freq(self):
    jan = Period('2000-01', 'M')
    day = Period('2012-01-01', 'D')
    assert not jan == day
    assert jan != day
    msg = 'Input has different freq=D from Period\\(freq=M\\)'
    with pytest.raises(IncompatibleFrequency, match=msg):
        jan < day
    with pytest.raises(IncompatibleFrequency, match=msg):
        jan <= day
    with pytest.raises(IncompatibleFrequency, match=msg):
        jan > day
    with pytest.raises(IncompatibleFrequency, match=msg):
        jan >= day