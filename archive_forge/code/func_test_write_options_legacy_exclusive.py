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
def test_write_options_legacy_exclusive(stream_fixture):
    with pytest.raises(ValueError, match='provide at most one of options and use_legacy_format'):
        stream_fixture.use_legacy_ipc_format = True
        stream_fixture.options = pa.ipc.IpcWriteOptions()
        stream_fixture.write_batches()