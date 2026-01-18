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
def test_read_until_regex_max_bytes(self: typing.Any):
    rs, ws = (yield self.make_iostream_pair())
    closed = Event()
    rs.set_close_callback(closed.set)
    try:
        fut = rs.read_until_regex(b'def', max_bytes=50)
        ws.write(b'abcdef')
        data = (yield fut)
        self.assertEqual(data, b'abcdef')
        fut = rs.read_until_regex(b'def', max_bytes=6)
        ws.write(b'abcdef')
        data = (yield fut)
        self.assertEqual(data, b'abcdef')
        with ExpectLog(gen_log, 'Unsatisfiable read', level=logging.INFO):
            rs.read_until_regex(b'def', max_bytes=5)
            ws.write(b'123456')
            yield closed.wait()
    finally:
        ws.close()
        rs.close()