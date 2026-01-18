from datetime import timedelta
import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas import (
def test_period_comparison_same_freq(self):
    jan = Period('2000-01', 'M')
    feb = Period('2000-02', 'M')
    assert not jan == feb
    assert jan != feb
    assert jan < feb
    assert jan <= feb
    assert not jan > feb
    assert not jan >= feb