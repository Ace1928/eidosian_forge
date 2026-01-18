from datetime import timedelta
import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas import (
@pytest.mark.parametrize('normalize', [True, False])
@pytest.mark.parametrize('n', [1, 2, 3, 4])
@pytest.mark.parametrize('offset, kwd_name', [(offsets.YearEnd, 'month'), (offsets.QuarterEnd, 'startingMonth'), (offsets.MonthEnd, None), (offsets.Week, 'weekday')])
def test_sub_n_gt_1_offsets(self, offset, kwd_name, n, normalize):
    kwds = {kwd_name: 3} if kwd_name is not None else {}
    p1_d = '19910905'
    p2_d = '19920406'
    p1 = Period(p1_d, freq=offset(n, normalize, **kwds))
    p2 = Period(p2_d, freq=offset(n, normalize, **kwds))
    expected = Period(p2_d, freq=p2.freq.base) - Period(p1_d, freq=p1.freq.base)
    assert p2 - p1 == expected