import socket
import typing  # noqa(F401)
from tornado.http1connection import HTTP1Connection
from tornado.httputil import HTTPMessageDelegate
from tornado.iostream import IOStream
from tornado.locks import Event
from tornado.netutil import add_accept_handler
from tornado.testing import AsyncTestCase, bind_unused_port, gen_test
def accept_callback(conn, addr):
    self.server_stream = IOStream(conn)
    self.addCleanup(self.server_stream.close)
    event.set()