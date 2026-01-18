import pytest
import decimal
import datetime
import pyarrow as pa
from pyarrow import fs
from pyarrow.tests import util
def test_orcfile_readwrite(tmpdir):
    from pyarrow import orc
    a = pa.array([1, None, 3, None])
    b = pa.array([None, 'Arrow', None, 'ORC'])
    table = pa.table({'int64': a, 'utf8': b})
    file = tmpdir.join('test.orc')
    orc.write_table(table, file)
    output_table = orc.read_table(file)
    assert table.equals(output_table)
    output_table = orc.read_table(file, [])
    assert 4 == output_table.num_rows
    assert 0 == output_table.num_columns
    output_table = orc.read_table(file, columns=['int64'])
    assert 4 == output_table.num_rows
    assert 1 == output_table.num_columns