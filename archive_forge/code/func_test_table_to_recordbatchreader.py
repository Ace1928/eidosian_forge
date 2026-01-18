from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_to_recordbatchreader():
    table = pa.Table.from_pydict({'x': [1, 2, 3]})
    reader = table.to_reader()
    assert table.schema == reader.schema
    assert table == reader.read_all()
    reader = table.to_reader(max_chunksize=2)
    assert reader.read_next_batch().num_rows == 2
    assert reader.read_next_batch().num_rows == 1