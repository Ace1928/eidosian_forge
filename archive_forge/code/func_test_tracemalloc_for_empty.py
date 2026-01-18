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
def test_tracemalloc_for_empty(self, table_type, dtype):
    with activated_tracemalloc():
        table = table_type()
        used = get_allocated_khash_memory()
        my_size = table.sizeof()
        assert used == my_size
        del table
        assert get_allocated_khash_memory() == 0