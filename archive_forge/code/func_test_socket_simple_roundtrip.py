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
def test_socket_simple_roundtrip(socket_fixture):
    socket_fixture.start_server(do_read_all=False)
    writer_batches = socket_fixture.write_batches()
    reader_schema, reader_batches = socket_fixture.stop_and_get_result()
    assert reader_schema.equals(writer_batches[0].schema)
    assert len(reader_batches) == len(writer_batches)
    for i, batch in enumerate(writer_batches):
        assert reader_batches[i].equals(batch)