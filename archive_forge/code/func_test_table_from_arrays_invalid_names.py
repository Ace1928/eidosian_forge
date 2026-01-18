from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_from_arrays_invalid_names():
    data = [pa.array(range(5)), pa.array([-10, -5, 0, 5, 10])]
    with pytest.raises(ValueError):
        pa.Table.from_arrays(data, names=['a', 'b', 'c'])
    with pytest.raises(ValueError):
        pa.Table.from_arrays(data, names=['a'])