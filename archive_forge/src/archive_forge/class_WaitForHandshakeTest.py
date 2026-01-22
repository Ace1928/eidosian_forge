from tornado.concurrent import Future
from tornado import gen
from tornado import netutil
from tornado.ioloop import IOLoop
from tornado.iostream import (
from tornado.httputil import HTTPHeaders
from tornado.locks import Condition, Event
from tornado.log import gen_log
from tornado.netutil import ssl_options_to_context, ssl_wrap_socket
from tornado.platform.asyncio import AddThreadSelectorEventLoop
from tornado.tcpserver import TCPServer
from tornado.testing import (
from tornado.test.util import (
from tornado.web import RequestHandler, Application
import asyncio
import errno
import hashlib
import logging
import os
import platform
import random
import socket
import ssl
import typing
from unittest import mock
import unittest
class WaitForHandshakeTest(AsyncTestCase):

    @gen.coroutine
    def connect_to_server(self, server_cls):
        server = client = None
        try:
            sock, port = bind_unused_port()
            server = server_cls(ssl_options=_server_ssl_options())
            server.add_socket(sock)
            ssl_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            with ignore_deprecation():
                ssl_ctx.options |= getattr(ssl, 'OP_NO_TLSv1_3', 0)
                client = SSLIOStream(socket.socket(), ssl_options=ssl_ctx)
            yield client.connect(('127.0.0.1', port))
            self.assertIsNotNone(client.socket.cipher())
        finally:
            if server is not None:
                server.stop()
            if client is not None:
                client.close()

    @gen_test
    def test_wait_for_handshake_future(self):
        test = self
        handshake_future = Future()

        class TestServer(TCPServer):

            def handle_stream(self, stream, address):
                test.assertIsNone(stream.socket.cipher())
                test.io_loop.spawn_callback(self.handle_connection, stream)

            @gen.coroutine
            def handle_connection(self, stream):
                yield stream.wait_for_handshake()
                handshake_future.set_result(None)
        yield self.connect_to_server(TestServer)
        yield handshake_future

    @gen_test
    def test_wait_for_handshake_already_waiting_error(self):
        test = self
        handshake_future = Future()

        class TestServer(TCPServer):

            @gen.coroutine
            def handle_stream(self, stream, address):
                fut = stream.wait_for_handshake()
                test.assertRaises(RuntimeError, stream.wait_for_handshake)
                yield fut
                handshake_future.set_result(None)
        yield self.connect_to_server(TestServer)
        yield handshake_future

    @gen_test
    def test_wait_for_handshake_already_connected(self):
        handshake_future = Future()

        class TestServer(TCPServer):

            @gen.coroutine
            def handle_stream(self, stream, address):
                yield stream.wait_for_handshake()
                yield stream.wait_for_handshake()
                handshake_future.set_result(None)
        yield self.connect_to_server(TestServer)
        yield handshake_future