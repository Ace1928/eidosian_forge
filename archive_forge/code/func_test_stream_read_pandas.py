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
@pytest.mark.pandas
def test_stream_read_pandas(stream_fixture):
    frames = [batch.to_pandas() for batch in stream_fixture.write_batches()]
    file_contents = stream_fixture.get_source()
    reader = pa.ipc.open_stream(file_contents)
    result = reader.read_pandas()
    expected = pd.concat(frames).reset_index(drop=True)
    assert_frame_equal(result, expected)