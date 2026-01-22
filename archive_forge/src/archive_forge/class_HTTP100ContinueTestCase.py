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
class HTTP100ContinueTestCase(AsyncHTTPTestCase):

    def respond_100(self, request):
        self.http1 = request.version.startswith('HTTP/1.')
        if not self.http1:
            request.connection.write_headers(ResponseStartLine('', 200, 'OK'), HTTPHeaders())
            request.connection.finish()
            return
        self.request = request
        fut = self.request.connection.stream.write(b'HTTP/1.1 100 CONTINUE\r\n\r\n')
        fut.add_done_callback(self.respond_200)

    def respond_200(self, fut):
        fut.result()
        fut = self.request.connection.stream.write(b'HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\nA')
        fut.add_done_callback(lambda f: self.request.connection.stream.close())

    def get_app(self):
        return self.respond_100

    def test_100_continue(self):
        res = self.fetch('/')
        if not self.http1:
            self.skipTest('requires HTTP/1.x')
        self.assertEqual(res.body, b'A')