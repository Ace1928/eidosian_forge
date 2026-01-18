import inspect
import pydoc
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_unknown_attribute(self):
    tdi = timedelta_range(start=0, periods=10, freq='1s')
    ser = Series(np.random.default_rng(2).normal(size=10), index=tdi)
    assert 'foo' not in ser.__dict__
    msg = "'Series' object has no attribute 'foo'"
    with pytest.raises(AttributeError, match=msg):
        ser.foo