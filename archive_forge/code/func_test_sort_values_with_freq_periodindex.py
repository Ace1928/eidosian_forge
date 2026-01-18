import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('freq', ['D', '2D', '4D'])
def test_sort_values_with_freq_periodindex(self, freq):
    idx = PeriodIndex(['2011-01-01', '2011-01-02', '2011-01-03'], freq=freq, name='idx')
    self.check_sort_values_with_freq(idx)