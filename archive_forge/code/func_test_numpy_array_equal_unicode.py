import copy
import numpy as np
import pytest
import pandas as pd
from pandas import Timestamp
import pandas._testing as tm
def test_numpy_array_equal_unicode():
    msg = 'numpy array are different\n\nnumpy array values are different \\(33\\.33333 %\\)\n\\[left\\]:  \\[á, à, ä\\]\n\\[right\\]: \\[á, à, å\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array(['á', 'à', 'ä']), np.array(['á', 'à', 'å']))