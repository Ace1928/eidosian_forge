from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
@pytest.mark.parametrize('data, klass', [((['', 'foo', 'bar'], [4.5, 5, None]), pa.array), (([[''], ['foo', 'bar']], [[4.5], [5.0, None]]), pa.chunked_array)])
def test_table_from_pydict_arrow_arrays(data, klass):
    data = OrderedDict([('strs', klass(data[0])), ('floats', klass(data[1]))])
    schema = pa.schema([('strs', pa.utf8()), ('floats', pa.float64())])
    table = pa.Table.from_pydict(data)
    assert table.num_columns == 2
    assert table.num_rows == 3
    assert table.schema == schema
    table = pa.Table.from_pydict(data, schema=schema)
    assert table.num_columns == 2
    assert table.num_rows == 3
    assert table.schema == schema
    schema = pa.schema([('strs', pa.utf8()), ('floats', pa.float32())])
    table = pa.Table.from_pydict(data, schema=schema)
    assert pa.types.is_float32(table.column('floats').type)
    assert table.num_columns == 2
    assert table.num_rows == 3
    assert table.schema == schema
    schema = pa.schema([('strs', pa.utf8()), ('floats', pa.timestamp('s'))])
    with pytest.raises((NotImplementedError, TypeError)):
        pa.Table.from_pydict(data, schema=schema)