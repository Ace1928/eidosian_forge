import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_pindex_fieldaccessor_nat(self):
    idx = PeriodIndex(['2011-01', '2011-02', 'NaT', '2012-03', '2012-04'], freq='D', name='name')
    exp = Index([2011, 2011, -1, 2012, 2012], dtype=np.int64, name='name')
    tm.assert_index_equal(idx.year, exp)
    exp = Index([1, 2, -1, 3, 4], dtype=np.int64, name='name')
    tm.assert_index_equal(idx.month, exp)