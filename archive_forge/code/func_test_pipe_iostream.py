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
def test_pipe_iostream(self):
    rs, ws = (yield self.make_iostream_pair())
    ws.write(b'hel')
    ws.write(b'lo world')
    data = (yield rs.read_until(b' '))
    self.assertEqual(data, b'hello ')
    data = (yield rs.read_bytes(3))
    self.assertEqual(data, b'wor')
    ws.close()
    data = (yield rs.read_until_close())
    self.assertEqual(data, b'ld')
    rs.close()