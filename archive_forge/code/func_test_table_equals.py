from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_equals():
    table = pa.Table.from_arrays([], names=[])
    assert table.equals(table)
    assert not table.equals(None)
    other = pa.Table.from_arrays([], names=[], metadata={'key': 'value'})
    assert not table.equals(other, check_metadata=True)
    assert table.equals(other)