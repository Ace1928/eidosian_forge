import errno
import os
import socket
import ssl
from tornado import gen
from tornado.log import app_log
from tornado.ioloop import IOLoop
from tornado.iostream import IOStream, SSLIOStream
from tornado.netutil import (
from tornado import process
from tornado.util import errno_from_exception
import typing
from typing import Union, Dict, Any, Iterable, Optional, Awaitable
def _handle_connection(self, connection: socket.socket, address: Any) -> None:
    if self.ssl_options is not None:
        assert ssl, 'Python 2.6+ and OpenSSL required for SSL'
        try:
            connection = ssl_wrap_socket(connection, self.ssl_options, server_side=True, do_handshake_on_connect=False)
        except ssl.SSLError as err:
            if err.args[0] == ssl.SSL_ERROR_EOF:
                return connection.close()
            else:
                raise
        except socket.error as err:
            if errno_from_exception(err) in (errno.ECONNABORTED, errno.EINVAL):
                return connection.close()
            else:
                raise
    try:
        if self.ssl_options is not None:
            stream = SSLIOStream(connection, max_buffer_size=self.max_buffer_size, read_chunk_size=self.read_chunk_size)
        else:
            stream = IOStream(connection, max_buffer_size=self.max_buffer_size, read_chunk_size=self.read_chunk_size)
        future = self.handle_stream(stream, address)
        if future is not None:
            IOLoop.current().add_future(gen.convert_yielded(future), lambda f: f.result())
    except Exception:
        app_log.error('Error in connection callback', exc_info=True)