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
class SocketStreamFixture(IpcFixture):

    def __init__(self):
        pass

    def start_server(self, do_read_all):
        self._server = StreamReaderServer()
        port = self._server.init(do_read_all)
        self._server.start()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect(('127.0.0.1', port))
        self.sink = self.get_sink()

    def stop_and_get_result(self):
        import struct
        self.sink.write(struct.pack('Q', 0))
        self.sink.flush()
        self._sock.close()
        self._server.join()
        return self._server.get_result()

    def get_sink(self):
        return self._sock.makefile(mode='wb')

    def _get_writer(self, sink, schema):
        return pa.RecordBatchStreamWriter(sink, schema)