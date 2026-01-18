from functools import partial
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_rolling_skew_eq_value_fperr(step):
    a = Series([1.1] * 15).rolling(window=10, step=step).skew()
    assert (a[a.index >= 9] == 0).all()
    assert a[a.index < 9].isna().all()