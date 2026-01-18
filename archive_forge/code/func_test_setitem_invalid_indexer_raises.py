import pickle
import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_ import (
from pandas.core.arrays.string_arrow import (
def test_setitem_invalid_indexer_raises():
    pa = pytest.importorskip('pyarrow')
    arr = ArrowStringArray(pa.array(list('abcde')))
    with pytest.raises(IndexError, match=None):
        arr[5] = 'foo'
    with pytest.raises(IndexError, match=None):
        arr[-6] = 'foo'
    with pytest.raises(IndexError, match=None):
        arr[[0, 5]] = 'foo'
    with pytest.raises(IndexError, match=None):
        arr[[0, -6]] = 'foo'
    with pytest.raises(IndexError, match=None):
        arr[[True, True, False]] = 'foo'
    with pytest.raises(ValueError, match=None):
        arr[[0, 1]] = ['foo', 'bar', 'baz']