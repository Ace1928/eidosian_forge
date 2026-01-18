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
def test_record_batch_reader_from_arrow_stream():

    class StreamWrapper:

        def __init__(self, batches):
            self.batches = batches

        def __arrow_c_stream__(self, requested_schema=None):
            reader = pa.RecordBatchReader.from_batches(self.batches[0].schema, self.batches)
            return reader.__arrow_c_stream__(requested_schema)
    data = [pa.record_batch([pa.array([1, 2, 3], type=pa.int64())], names=['a']), pa.record_batch([pa.array([4, 5, 6], type=pa.int64())], names=['a'])]
    wrapper = StreamWrapper(data)
    expected = pa.Table.from_batches(data)
    reader = pa.RecordBatchReader.from_stream(expected)
    assert reader.read_all() == expected
    reader = pa.RecordBatchReader.from_stream(wrapper)
    assert reader.read_all() == expected
    reader = pa.RecordBatchReader.from_stream(wrapper, schema=data[0].schema)
    assert reader.read_all() == expected
    with pytest.raises(NotImplementedError):
        pa.RecordBatchReader.from_stream(wrapper, schema=pa.schema([pa.field('a', pa.int32())]))
    with pytest.raises(TypeError):
        pa.RecordBatchReader.from_stream(data[0]['a'])
    with pytest.raises(TypeError):
        pa.RecordBatchReader.from_stream(expected, schema=data[0])