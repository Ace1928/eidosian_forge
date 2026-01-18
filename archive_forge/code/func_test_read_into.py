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
def test_read_into(self: typing.Any):
    rs, ws = (yield self.make_iostream_pair())

    def sleep_some():
        self.io_loop.run_sync(lambda: gen.sleep(0.05))
    try:
        buf = bytearray(10)
        fut = rs.read_into(buf)
        ws.write(b'hello')
        yield gen.sleep(0.05)
        self.assertTrue(rs.reading())
        ws.write(b'world!!')
        data = (yield fut)
        self.assertFalse(rs.reading())
        self.assertEqual(data, 10)
        self.assertEqual(bytes(buf), b'helloworld')
        fut = rs.read_into(buf)
        yield gen.sleep(0.05)
        self.assertTrue(rs.reading())
        ws.write(b'1234567890')
        data = (yield fut)
        self.assertFalse(rs.reading())
        self.assertEqual(data, 10)
        self.assertEqual(bytes(buf), b'!!12345678')
        buf = bytearray(4)
        ws.write(b'abcdefghi')
        data = (yield rs.read_into(buf))
        self.assertEqual(data, 4)
        self.assertEqual(bytes(buf), b'90ab')
        data = (yield rs.read_bytes(7))
        self.assertEqual(data, b'cdefghi')
    finally:
        ws.close()
        rs.close()