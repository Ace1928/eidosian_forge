import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('idx,expected', [(PeriodIndex(['2011-01-01', '2011-01-03', '2011-01-05', '2011-01-02', '2011-01-01'], freq='D', name='idx1'), PeriodIndex(['2011-01-01', '2011-01-01', '2011-01-02', '2011-01-03', '2011-01-05'], freq='D', name='idx1')), (PeriodIndex(['2011-01-01', '2011-01-03', '2011-01-05', '2011-01-02', '2011-01-01'], freq='D', name='idx2'), PeriodIndex(['2011-01-01', '2011-01-01', '2011-01-02', '2011-01-03', '2011-01-05'], freq='D', name='idx2')), (PeriodIndex([NaT, '2011-01-03', '2011-01-05', '2011-01-02', NaT], freq='D', name='idx3'), PeriodIndex([NaT, NaT, '2011-01-02', '2011-01-03', '2011-01-05'], freq='D', name='idx3')), (PeriodIndex(['2011', '2013', '2015', '2012', '2011'], name='pidx', freq='Y'), PeriodIndex(['2011', '2011', '2012', '2013', '2015'], name='pidx', freq='Y')), (Index([2011, 2013, 2015, 2012, 2011], name='idx'), Index([2011, 2011, 2012, 2013, 2015], name='idx'))])
def test_sort_values_without_freq_periodindex(self, idx, expected):
    self.check_sort_values_without_freq(idx, expected)