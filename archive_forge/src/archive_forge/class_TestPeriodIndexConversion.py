import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestPeriodIndexConversion:

    def test_tolist(self):
        index = period_range(freq='Y', start='1/1/2001', end='12/1/2009')
        rs = index.tolist()
        for x in rs:
            assert isinstance(x, Period)
        recon = PeriodIndex(rs)
        tm.assert_index_equal(index, recon)