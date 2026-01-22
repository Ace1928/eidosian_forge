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
class StreamFormatFixture(IpcFixture):
    use_legacy_ipc_format = False
    options = None
    is_file = False

    def _get_writer(self, sink, schema):
        return pa.ipc.new_stream(sink, schema, use_legacy_format=self.use_legacy_ipc_format, options=self.options)