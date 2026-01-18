import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('freq', ['D', 'h'])
def test_sort_values_with_freq_timedeltaindex(self, freq):
    idx = timedelta_range(start=f'1{freq}', periods=3, freq=freq).rename('idx')
    self.check_sort_values_with_freq(idx)