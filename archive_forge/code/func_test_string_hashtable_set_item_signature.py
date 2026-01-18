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
def test_string_hashtable_set_item_signature(self):
    tbl = ht.StringHashTable()
    tbl.set_item('key', 1)
    assert tbl.get_item('key') == 1
    with pytest.raises(TypeError, match="'key' has incorrect type"):
        tbl.set_item(4, 6)
    with pytest.raises(TypeError, match="'val' has incorrect type"):
        tbl.get_item(4)