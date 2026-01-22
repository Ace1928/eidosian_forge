from tornado import gen, netutil
from tornado.escape import (
from tornado.http1connection import HTTP1Connection
from tornado.httpclient import HTTPError
from tornado.httpserver import HTTPServer
from tornado.httputil import (
from tornado.iostream import IOStream
from tornado.locks import Event
from tornado.log import gen_log, app_log
from tornado.netutil import ssl_options_to_context
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.testing import (
from tornado.test.util import skipOnTravis
from tornado.web import Application, RequestHandler, stream_request_body
from contextlib import closing
import datetime
import gzip
import logging
import os
import shutil
import socket
import ssl
import sys
import tempfile
import textwrap
import unittest
import urllib.parse
from io import BytesIO
import typing
class KeepAliveTest(AsyncHTTPTestCase):
    """Tests various scenarios for HTTP 1.1 keep-alive support.

    These tests don't use AsyncHTTPClient because we want to control
    connection reuse and closing.
    """

    def get_app(self):

        class HelloHandler(RequestHandler):

            def get(self):
                self.finish('Hello world')

            def post(self):
                self.finish('Hello world')

        class LargeHandler(RequestHandler):

            def get(self):
                self.write(''.join((chr(i % 256) * 1024 for i in range(512))))

        class TransferEncodingChunkedHandler(RequestHandler):

            @gen.coroutine
            def head(self):
                self.write('Hello world')
                yield self.flush()

        class FinishOnCloseHandler(RequestHandler):

            def initialize(self, cleanup_event):
                self.cleanup_event = cleanup_event

            @gen.coroutine
            def get(self):
                self.flush()
                yield self.cleanup_event.wait()

            def on_connection_close(self):
                self.finish('closed')
        self.cleanup_event = Event()
        return Application([('/', HelloHandler), ('/large', LargeHandler), ('/chunked', TransferEncodingChunkedHandler), ('/finish_on_close', FinishOnCloseHandler, dict(cleanup_event=self.cleanup_event))])

    def setUp(self):
        super().setUp()
        self.http_version = b'HTTP/1.1'

    def tearDown(self):
        self.io_loop.add_timeout(datetime.timedelta(seconds=0.001), self.stop)
        self.wait()
        if hasattr(self, 'stream'):
            self.stream.close()
        super().tearDown()

    @gen.coroutine
    def connect(self):
        self.stream = IOStream(socket.socket())
        yield self.stream.connect(('127.0.0.1', self.get_http_port()))

    @gen.coroutine
    def read_headers(self):
        first_line = (yield self.stream.read_until(b'\r\n'))
        self.assertTrue(first_line.startswith(b'HTTP/1.1 200'), first_line)
        header_bytes = (yield self.stream.read_until(b'\r\n\r\n'))
        headers = HTTPHeaders.parse(header_bytes.decode('latin1'))
        raise gen.Return(headers)

    @gen.coroutine
    def read_response(self):
        self.headers = (yield self.read_headers())
        body = (yield self.stream.read_bytes(int(self.headers['Content-Length'])))
        self.assertEqual(b'Hello world', body)

    def close(self):
        self.stream.close()
        del self.stream

    @gen_test
    def test_two_requests(self):
        yield self.connect()
        self.stream.write(b'GET / HTTP/1.1\r\n\r\n')
        yield self.read_response()
        self.stream.write(b'GET / HTTP/1.1\r\n\r\n')
        yield self.read_response()
        self.close()

    @gen_test
    def test_request_close(self):
        yield self.connect()
        self.stream.write(b'GET / HTTP/1.1\r\nConnection: close\r\n\r\n')
        yield self.read_response()
        data = (yield self.stream.read_until_close())
        self.assertTrue(not data)
        self.assertEqual(self.headers['Connection'], 'close')
        self.close()

    @gen_test
    def test_http10(self):
        self.http_version = b'HTTP/1.0'
        yield self.connect()
        self.stream.write(b'GET / HTTP/1.0\r\n\r\n')
        yield self.read_response()
        data = (yield self.stream.read_until_close())
        self.assertTrue(not data)
        self.assertTrue('Connection' not in self.headers)
        self.close()

    @gen_test
    def test_http10_keepalive(self):
        self.http_version = b'HTTP/1.0'
        yield self.connect()
        self.stream.write(b'GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n')
        yield self.read_response()
        self.assertEqual(self.headers['Connection'], 'Keep-Alive')
        self.stream.write(b'GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n')
        yield self.read_response()
        self.assertEqual(self.headers['Connection'], 'Keep-Alive')
        self.close()

    @gen_test
    def test_http10_keepalive_extra_crlf(self):
        self.http_version = b'HTTP/1.0'
        yield self.connect()
        self.stream.write(b'GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n\r\n')
        yield self.read_response()
        self.assertEqual(self.headers['Connection'], 'Keep-Alive')
        self.stream.write(b'GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n')
        yield self.read_response()
        self.assertEqual(self.headers['Connection'], 'Keep-Alive')
        self.close()

    @gen_test
    def test_pipelined_requests(self):
        yield self.connect()
        self.stream.write(b'GET / HTTP/1.1\r\n\r\nGET / HTTP/1.1\r\n\r\n')
        yield self.read_response()
        yield self.read_response()
        self.close()

    @gen_test
    def test_pipelined_cancel(self):
        yield self.connect()
        self.stream.write(b'GET / HTTP/1.1\r\n\r\nGET / HTTP/1.1\r\n\r\n')
        yield self.read_response()
        self.close()

    @gen_test
    def test_cancel_during_download(self):
        yield self.connect()
        self.stream.write(b'GET /large HTTP/1.1\r\n\r\n')
        yield self.read_headers()
        yield self.stream.read_bytes(1024)
        self.close()

    @gen_test
    def test_finish_while_closed(self):
        yield self.connect()
        self.stream.write(b'GET /finish_on_close HTTP/1.1\r\n\r\n')
        yield self.read_headers()
        self.close()
        self.cleanup_event.set()

    @gen_test
    def test_keepalive_chunked(self):
        self.http_version = b'HTTP/1.0'
        yield self.connect()
        self.stream.write(b'POST / HTTP/1.0\r\nConnection: keep-alive\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n')
        yield self.read_response()
        self.assertEqual(self.headers['Connection'], 'Keep-Alive')
        self.stream.write(b'GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n')
        yield self.read_response()
        self.assertEqual(self.headers['Connection'], 'Keep-Alive')
        self.close()

    @gen_test
    def test_keepalive_chunked_head_no_body(self):
        yield self.connect()
        self.stream.write(b'HEAD /chunked HTTP/1.1\r\n\r\n')
        yield self.read_headers()
        self.stream.write(b'HEAD /chunked HTTP/1.1\r\n\r\n')
        yield self.read_headers()
        self.close()