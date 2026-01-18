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
def test_write_empty_ipc_file():
    schema = pa.schema([('field', pa.int64())])
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, schema):
        pass
    buf = sink.getvalue()
    with pa.RecordBatchFileReader(pa.BufferReader(buf)) as reader:
        table = reader.read_all()
    assert len(table) == 0
    assert table.schema.equals(schema)