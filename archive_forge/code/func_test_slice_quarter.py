from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_slice_quarter(self):
    dti = date_range(freq='D', start=datetime(2000, 6, 1), periods=500)
    s = Series(np.arange(len(dti)), index=dti)
    assert len(s['2001Q1']) == 90
    df = DataFrame(np.random.default_rng(2).random((len(dti), 5)), index=dti)
    assert len(df.loc['1Q01']) == 90