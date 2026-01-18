import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_sort_values_without_freq_periodindex_nat(self):
    idx = PeriodIndex(['2011', '2013', 'NaT', '2011'], name='pidx', freq='D')
    expected = PeriodIndex(['NaT', '2011', '2011', '2013'], name='pidx', freq='D')
    ordered = idx.sort_values(na_position='first')
    tm.assert_index_equal(ordered, expected)
    check_freq_nonmonotonic(ordered, idx)
    ordered = idx.sort_values(ascending=False)
    tm.assert_index_equal(ordered, expected[::-1])
    check_freq_nonmonotonic(ordered, idx)