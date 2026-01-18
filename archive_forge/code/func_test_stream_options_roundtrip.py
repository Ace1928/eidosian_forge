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
@pytest.mark.parametrize('options', [pa.ipc.IpcWriteOptions(), pa.ipc.IpcWriteOptions(allow_64bit=True), pa.ipc.IpcWriteOptions(use_legacy_format=True), pa.ipc.IpcWriteOptions(metadata_version=pa.ipc.MetadataVersion.V4), pa.ipc.IpcWriteOptions(use_legacy_format=True, metadata_version=pa.ipc.MetadataVersion.V4)])
def test_stream_options_roundtrip(stream_fixture, options):
    stream_fixture.use_legacy_ipc_format = None
    stream_fixture.options = options
    batches = stream_fixture.write_batches()
    file_contents = pa.BufferReader(stream_fixture.get_source())
    message = pa.ipc.read_message(stream_fixture.get_source())
    assert message.metadata_version == options.metadata_version
    reader = pa.ipc.open_stream(file_contents)
    assert reader.schema.equals(batches[0].schema)
    total = 0
    for i, next_batch in enumerate(reader):
        assert next_batch.equals(batches[i])
        total += 1
    assert total == len(batches)
    with pytest.raises(StopIteration):
        reader.read_next_batch()