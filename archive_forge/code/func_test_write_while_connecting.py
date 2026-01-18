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
def test_write_while_connecting(self: typing.Any):
    stream = self._make_client_iostream()
    connect_fut = stream.connect(('127.0.0.1', self.get_http_port()))
    write_fut = stream.write(b'GET / HTTP/1.0\r\nConnection: close\r\n\r\n')
    self.assertFalse(connect_fut.done())
    it = gen.WaitIterator(connect_fut, write_fut)
    resolved_order = []
    while not it.done():
        yield it.next()
        resolved_order.append(it.current_future)
    self.assertEqual(resolved_order, [connect_fut, write_fut])
    data = (yield stream.read_until_close())
    self.assertTrue(data.endswith(b'Hello'))
    stream.close()