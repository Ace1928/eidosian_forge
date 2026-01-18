from collections import UserList
import io
import pathlib
import pytest
import socket
import threading
import weakref
import numpy as np
import pyarrow as pa
from pyarrow.tests.util import changed_environ, invoke_script
def test_read_options_included_fields(stream_fixture):
    options1 = pa.ipc.IpcReadOptions()
    options2 = pa.ipc.IpcReadOptions(included_fields=[1])
    table = pa.Table.from_arrays([pa.array(['foo', 'bar', 'baz', 'qux']), pa.array([1, 2, 3, 4])], names=['a', 'b'])
    with stream_fixture._get_writer(stream_fixture.sink, table.schema) as wr:
        wr.write_table(table)
    source = stream_fixture.get_source()
    reader1 = pa.ipc.open_stream(source, options=options1)
    reader2 = pa.ipc.open_stream(source, options=options2, memory_pool=pa.system_memory_pool())
    result1 = reader1.read_all()
    result2 = reader2.read_all()
    assert result1.num_columns == 2
    assert result2.num_columns == 1
    expected = pa.Table.from_arrays([pa.array([1, 2, 3, 4])], names=['b'])
    assert result2 == expected
    assert result1 == table