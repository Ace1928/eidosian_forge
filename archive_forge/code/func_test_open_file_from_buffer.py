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
def test_open_file_from_buffer(file_fixture):
    file_fixture.write_batches()
    source = file_fixture.get_source()
    reader1 = pa.ipc.open_file(source)
    reader2 = pa.ipc.open_file(pa.BufferReader(source))
    reader3 = pa.RecordBatchFileReader(source)
    result1 = reader1.read_all()
    result2 = reader2.read_all()
    result3 = reader3.read_all()
    assert result1.equals(result2)
    assert result1.equals(result3)
    st1 = reader1.stats
    assert st1.num_messages == 6
    assert st1.num_record_batches == 5
    assert reader2.stats == st1
    assert reader3.stats == st1