import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_unstack_preserves_object():
    mi = MultiIndex.from_product([['bar', 'foo'], ['one', 'two']])
    ser = Series(np.arange(4.0), index=mi, dtype=object)
    res1 = ser.unstack()
    assert (res1.dtypes == object).all()
    res2 = ser.unstack(level=0)
    assert (res2.dtypes == object).all()