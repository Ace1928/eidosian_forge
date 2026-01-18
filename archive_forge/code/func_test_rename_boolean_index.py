from collections import ChainMap
import inspect
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_boolean_index(self):
    df = DataFrame(np.arange(15).reshape(3, 5), columns=[False, True, 2, 3, 4])
    mapper = {0: 'foo', 1: 'bar', 2: 'bah'}
    res = df.rename(index=mapper)
    exp = DataFrame(np.arange(15).reshape(3, 5), columns=[False, True, 2, 3, 4], index=['foo', 'bar', 'bah'])
    tm.assert_frame_equal(res, exp)