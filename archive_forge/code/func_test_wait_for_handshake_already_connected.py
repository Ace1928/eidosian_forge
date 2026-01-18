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