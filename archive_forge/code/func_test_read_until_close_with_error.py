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
def test_read_until_close_with_error(self: typing.Any):
    server, client = (yield self.make_iostream_pair())
    try:
        with mock.patch('tornado.iostream.BaseIOStream._try_inline_read', side_effect=IOError('boom')):
            with self.assertRaisesRegex(IOError, 'boom'):
                client.read_until_close()
    finally:
        server.close()
        client.close()