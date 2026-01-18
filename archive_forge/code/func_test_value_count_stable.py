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
def test_value_count_stable(self, dtype, writable):
    values = np.array([2, 1, 5, 22, 3, -1, 8]).astype(dtype)
    values.flags.writeable = writable
    keys, counts, _ = ht.value_count(values, False)
    tm.assert_numpy_array_equal(keys, values)
    assert np.all(counts == 1)