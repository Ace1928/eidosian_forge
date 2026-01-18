from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_basics():
    data = [pa.array(range(5), type='int64'), pa.array([-10, -5, 0, 5, 10], type='int64')]
    table = pa.table(data, names=('a', 'b'))
    table.validate()
    assert len(table) == 5
    assert table.num_rows == 5
    assert table.num_columns == 2
    assert table.shape == (5, 2)
    assert table.get_total_buffer_size() == 2 * (5 * 8)
    assert table.nbytes == 2 * (5 * 8)
    assert sys.getsizeof(table) >= object.__sizeof__(table) + table.get_total_buffer_size()
    pydict = table.to_pydict()
    assert pydict == OrderedDict([('a', [0, 1, 2, 3, 4]), ('b', [-10, -5, 0, 5, 10])])
    assert isinstance(pydict, dict)
    columns = []
    for col in table.itercolumns():
        columns.append(col)
        for chunk in col.iterchunks():
            assert chunk is not None
        with pytest.raises(IndexError):
            col.chunk(-1)
        with pytest.raises(IndexError):
            col.chunk(col.num_chunks)
    assert table.columns == columns
    assert table == pa.table(columns, names=table.column_names)
    assert table != pa.table(columns[1:], names=table.column_names[1:])
    assert table != columns
    wr = weakref.ref(table)
    assert wr() is not None
    del table
    assert wr() is None