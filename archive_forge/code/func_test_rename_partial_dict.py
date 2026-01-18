from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_partial_dict(self):
    ser = Series(np.arange(4), index=['a', 'b', 'c', 'd'], dtype='int64')
    renamed = ser.rename({'b': 'foo', 'd': 'bar'})
    tm.assert_index_equal(renamed.index, Index(['a', 'foo', 'c', 'bar']))