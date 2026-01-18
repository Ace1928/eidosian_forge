from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_from_arrays_preserves_column_metadata():
    arr0 = pa.array([1, 2])
    arr1 = pa.array([3, 4])
    field0 = pa.field('field1', pa.int64(), metadata=dict(a='A', b='B'))
    field1 = pa.field('field2', pa.int64(), nullable=False)
    table = pa.Table.from_arrays([arr0, arr1], schema=pa.schema([field0, field1]))
    assert b'a' in table.field(0).metadata
    assert table.field(1).nullable is False