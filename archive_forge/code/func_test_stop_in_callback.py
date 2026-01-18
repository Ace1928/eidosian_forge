import socket
import subprocess
import sys
import textwrap
import unittest
from tornado import gen
from tornado.iostream import IOStream
from tornado.log import app_log
from tornado.tcpserver import TCPServer
from tornado.test.util import skipIfNonUnix
from tornado.testing import AsyncTestCase, ExpectLog, bind_unused_port, gen_test
from typing import Tuple
@gen_test
def test_stop_in_callback(self):

    class TestServer(TCPServer):

        @gen.coroutine
        def handle_stream(self, stream, address):
            server.stop()
            yield stream.read_until_close()
    sock, port = bind_unused_port()
    server = TestServer()
    server.add_socket(sock)
    server_addr = ('localhost', port)
    N = 40
    clients = [IOStream(socket.socket()) for i in range(N)]
    connected_clients = []

    @gen.coroutine
    def connect(c):
        try:
            yield c.connect(server_addr)
        except EnvironmentError:
            pass
        else:
            connected_clients.append(c)
    yield [connect(c) for c in clients]
    self.assertGreater(len(connected_clients), 0, 'all clients failed connecting')
    try:
        if len(connected_clients) == N:
            self.skipTest('at least one client should fail connecting for the test to be meaningful')
    finally:
        for c in connected_clients:
            c.close()