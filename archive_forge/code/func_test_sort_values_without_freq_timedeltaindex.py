import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_sort_values_without_freq_timedeltaindex(self):
    idx = TimedeltaIndex(['1 hour', '3 hour', '5 hour', '2 hour ', '1 hour'], name='idx1')
    expected = TimedeltaIndex(['1 hour', '1 hour', '2 hour', '3 hour', '5 hour'], name='idx1')
    self.check_sort_values_without_freq(idx, expected)