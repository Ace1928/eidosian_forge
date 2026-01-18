import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('idx', [PeriodIndex(['2011', '2012', '2013'], name='pidx', freq='Y'), Index([2011, 2012, 2013], name='idx')])
def test_sort_values_with_freq_periodindex2(self, idx):
    self.check_sort_values_with_freq(idx)