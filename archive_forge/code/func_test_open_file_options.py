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
@pytest.mark.parametrize('options', [pa.ipc.IpcReadOptions(), pa.ipc.IpcReadOptions(use_threads=False)])
def test_open_file_options(file_fixture, options):
    file_fixture.write_batches()
    source = file_fixture.get_source()
    reader = pa.ipc.open_file(source, options=options)
    reader.read_all()
    st = reader.stats
    assert st.num_messages == 6
    assert st.num_record_batches == 5