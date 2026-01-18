from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_take_non_consecutive():
    table = pa.table([pa.array([1, 2, 3, None, 5]), pa.array(['a', 'b', 'c', 'd', 'e'])], ['f1', 'f2'])
    result_non_consecutive = pa.table([pa.array([2, None]), pa.array(['b', 'd'])], ['f1', 'f2'])
    assert table.take(pa.array([1, 3])).equals(result_non_consecutive)