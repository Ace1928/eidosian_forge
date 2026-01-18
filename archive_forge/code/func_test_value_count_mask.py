from collections.abc import Generator
from contextlib import contextmanager
import re
import struct
import tracemalloc
import numpy as np
import pytest
from pandas._libs import hashtable as ht
import pandas as pd
import pandas._testing as tm
from pandas.core.algorithms import isin
def test_value_count_mask(self, dtype):
    if dtype == np.object_:
        pytest.skip('mask not implemented for object dtype')
    values = np.array([1] * 5, dtype=dtype)
    mask = np.zeros((5,), dtype=np.bool_)
    mask[1] = True
    mask[4] = True
    keys, counts, na_counter = ht.value_count(values, False, mask=mask)
    assert len(keys) == 2
    assert na_counter == 2