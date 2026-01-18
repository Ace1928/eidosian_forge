from datetime import timedelta
import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas import (
@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_sub_n_gt_1_ticks(self, tick_classes, n):
    p1 = Period('19910905', freq=tick_classes(n))
    p2 = Period('19920406', freq=tick_classes(n))
    expected = Period(str(p2), freq=p2.freq.base) - Period(str(p1), freq=p1.freq.base)
    assert p2 - p1 == expected