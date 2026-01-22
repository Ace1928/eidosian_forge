import collections
from contextlib import closing
import errno
import logging
import os
import re
import socket
import ssl
import sys
import typing  # noqa: F401
from tornado.escape import to_unicode, utf8
from tornado import gen, version
from tornado.httpclient import AsyncHTTPClient
from tornado.httputil import HTTPHeaders, ResponseStartLine
from tornado.ioloop import IOLoop
from tornado.iostream import UnsatisfiableReadError
from tornado.locks import Event
from tornado.log import gen_log
from tornado.netutil import Resolver, bind_sockets
from tornado.simple_httpclient import (
from tornado.test.httpclient_test import (
from tornado.test import httpclient_test
from tornado.testing import (
from tornado.test.util import skipOnTravis, skipIfNoIPv6, refusing_port
from tornado.web import RequestHandler, Application, url, stream_request_body
class MaxBufferSizeTest(AsyncHTTPTestCase):

    def get_app(self):

        class LargeBody(RequestHandler):

            def get(self):
                self.write('a' * 1024 * 100)
        return Application([('/large', LargeBody)])

    def get_http_client(self):
        return SimpleAsyncHTTPClient(max_body_size=1024 * 100, max_buffer_size=1024 * 64)

    def test_large_body(self):
        response = self.fetch('/large')
        response.rethrow()
        self.assertEqual(response.body, b'a' * 1024 * 100)