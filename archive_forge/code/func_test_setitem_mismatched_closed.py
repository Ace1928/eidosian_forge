import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_setitem_mismatched_closed(self):
    arr = IntervalArray.from_breaks(range(4))
    orig = arr.copy()
    other = arr.set_closed('both')
    msg = "'value.closed' is 'both', expected 'right'"
    with pytest.raises(ValueError, match=msg):
        arr[0] = other[0]
    with pytest.raises(ValueError, match=msg):
        arr[:1] = other[:1]
    with pytest.raises(ValueError, match=msg):
        arr[:0] = other[:0]
    with pytest.raises(ValueError, match=msg):
        arr[:] = other[::-1]
    with pytest.raises(ValueError, match=msg):
        arr[:] = list(other[::-1])
    with pytest.raises(ValueError, match=msg):
        arr[:] = other[::-1].astype(object)
    with pytest.raises(ValueError, match=msg):
        arr[:] = other[::-1].astype('category')
    arr[:0] = []
    tm.assert_interval_array_equal(arr, orig)