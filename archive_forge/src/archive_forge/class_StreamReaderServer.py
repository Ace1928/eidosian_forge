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
class StreamReaderServer(threading.Thread):

    def init(self, do_read_all):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(('127.0.0.1', 0))
        self._sock.listen(1)
        host, port = self._sock.getsockname()
        self._do_read_all = do_read_all
        self._schema = None
        self._batches = []
        self._table = None
        return port

    def run(self):
        connection, client_address = self._sock.accept()
        try:
            source = connection.makefile(mode='rb')
            reader = pa.ipc.open_stream(source)
            self._schema = reader.schema
            if self._do_read_all:
                self._table = reader.read_all()
            else:
                for i, batch in enumerate(reader):
                    self._batches.append(batch)
        finally:
            connection.close()
            self._sock.close()

    def get_result(self):
        return (self._schema, self._table if self._do_read_all else self._batches)