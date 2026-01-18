from datetime import timedelta
import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas import (
def test_period_comparison_same_period_different_object(self):
    left = Period('2000-01', 'M')
    right = Period('2000-01', 'M')
    assert left == right
    assert left >= right
    assert left <= right
    assert not left < right
    assert not left > right