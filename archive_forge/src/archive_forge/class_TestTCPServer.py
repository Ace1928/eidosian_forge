from contextlib import closing
import getpass
import os
import socket
import unittest
from tornado.concurrent import Future
from tornado.netutil import bind_sockets, Resolver
from tornado.queues import Queue
from tornado.tcpclient import TCPClient, _Connector
from tornado.tcpserver import TCPServer
from tornado.testing import AsyncTestCase, gen_test
from tornado.test.util import skipIfNoIPv6, refusing_port, skipIfNonUnix
from tornado.gen import TimeoutError
import typing
class TestTCPServer(TCPServer):

    def __init__(self, family):
        super().__init__()
        self.streams = []
        self.queue = Queue()
        sockets = bind_sockets(0, 'localhost', family)
        self.add_sockets(sockets)
        self.port = sockets[0].getsockname()[1]

    def handle_stream(self, stream, address):
        self.streams.append(stream)
        self.queue.put(stream)

    def stop(self):
        super().stop()
        for stream in self.streams:
            stream.close()