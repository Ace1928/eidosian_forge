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
class HostnameMappingTestCase(AsyncHTTPTestCase):

    def setUp(self):
        super().setUp()
        self.http_client = SimpleAsyncHTTPClient(hostname_mapping={'www.example.com': '127.0.0.1', ('foo.example.com', 8000): ('127.0.0.1', self.get_http_port())})

    def get_app(self):
        return Application([url('/hello', HelloWorldHandler)])

    def test_hostname_mapping(self):
        response = self.fetch('http://www.example.com:%d/hello' % self.get_http_port())
        response.rethrow()
        self.assertEqual(response.body, b'Hello world!')

    def test_port_mapping(self):
        response = self.fetch('http://foo.example.com:8000/hello')
        response.rethrow()
        self.assertEqual(response.body, b'Hello world!')