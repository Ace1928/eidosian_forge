from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_cast_invalid():
    table = pa.table({'a': [None, 1], 'b': [None, True]})
    new_schema = pa.schema([pa.field('a', 'int64', nullable=True), pa.field('b', 'bool', nullable=False)])
    with pytest.raises(ValueError):
        table.cast(new_schema)
    table = pa.table({'a': [None, 1], 'b': [False, True]})
    assert table.cast(new_schema).schema == new_schema