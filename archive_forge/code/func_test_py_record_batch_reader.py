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
def test_py_record_batch_reader():

    def make_schema():
        return pa.schema([('field', pa.int64())])

    def make_batches():
        schema = make_schema()
        batch1 = pa.record_batch([[1, 2, 3]], schema=schema)
        batch2 = pa.record_batch([[4, 5]], schema=schema)
        return [batch1, batch2]
    batches = UserList(make_batches())
    wr = weakref.ref(batches)
    with pa.RecordBatchReader.from_batches(make_schema(), batches) as reader:
        batches = None
        assert wr() is not None
        assert list(reader) == make_batches()
        assert wr() is None
    batches = iter(UserList(make_batches()))
    wr = weakref.ref(batches)
    with pa.RecordBatchReader.from_batches(make_schema(), batches) as reader:
        batches = None
        assert wr() is not None
        assert list(reader) == make_batches()
        assert wr() is None
    batches = make_batches()
    with pytest.raises(TypeError):
        reader = pa.RecordBatchReader.from_batches([('field', pa.int64())], batches)
        pass
    with pytest.raises(TypeError):
        reader = pa.RecordBatchReader.from_batches(None, batches)
        pass