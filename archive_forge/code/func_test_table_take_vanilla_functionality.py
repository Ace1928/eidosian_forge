from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_take_vanilla_functionality():
    table = pa.table([pa.array([1, 2, 3, None, 5]), pa.array(['a', 'b', 'c', 'd', 'e'])], ['f1', 'f2'])
    assert table.take(pa.array([2, 3])).equals(table.slice(2, 2))