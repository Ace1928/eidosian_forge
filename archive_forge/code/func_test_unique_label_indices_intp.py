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
def test_unique_label_indices_intp(writable):
    keys = np.array([1, 2, 2, 2, 1, 3], dtype=np.intp)
    keys.flags.writeable = writable
    result = ht.unique_label_indices(keys)
    expected = np.array([0, 1, 5], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)