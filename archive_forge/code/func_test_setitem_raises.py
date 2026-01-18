from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_setitem_raises(self, arr1d):
    arr = arr1d[:10]
    val = arr[0]
    with pytest.raises(IndexError, match='index 12 is out of bounds'):
        arr[12] = val
    with pytest.raises(TypeError, match="value should be a.* 'object'"):
        arr[0] = object()
    msg = 'cannot set using a list-like indexer with a different length'
    with pytest.raises(ValueError, match=msg):
        arr[[]] = [arr[1]]
    msg = 'cannot set using a slice indexer with a different length than'
    with pytest.raises(ValueError, match=msg):
        arr[1:1] = arr[:3]