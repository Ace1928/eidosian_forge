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
class HTTPConnectionTest(AsyncHTTPTestCase):

    def get_handlers(self):
        return [('/multipart', MultipartTestHandler), ('/hello', HelloWorldRequestHandler)]

    def get_app(self):
        return Application(self.get_handlers())

    def raw_fetch(self, headers, body, newline=b'\r\n'):
        with closing(IOStream(socket.socket())) as stream:
            self.io_loop.run_sync(lambda: stream.connect(('127.0.0.1', self.get_http_port())))
            stream.write(newline.join(headers + [utf8('Content-Length: %d' % len(body))]) + newline + newline + body)
            start_line, headers, body = self.io_loop.run_sync(lambda: read_stream_body(stream))
            return body

    def test_multipart_form(self):
        response = self.raw_fetch([b'POST /multipart HTTP/1.0', b'Content-Type: multipart/form-data; boundary=1234567890', b'X-Header-encoding-test: \xe9'], b'\r\n'.join([b'Content-Disposition: form-data; name=argument', b'', 'á'.encode('utf-8'), b'--1234567890', 'Content-Disposition: form-data; name="files"; filename="ó"'.encode('utf8'), b'', 'ú'.encode('utf-8'), b'--1234567890--', b'']))
        data = json_decode(response)
        self.assertEqual('é', data['header'])
        self.assertEqual('á', data['argument'])
        self.assertEqual('ó', data['filename'])
        self.assertEqual('ú', data['filebody'])

    def test_newlines(self):
        for newline in (b'\r\n', b'\n'):
            response = self.raw_fetch([b'GET /hello HTTP/1.0'], b'', newline=newline)
            self.assertEqual(response, b'Hello world')

    @gen_test
    def test_100_continue(self):
        stream = IOStream(socket.socket())
        yield stream.connect(('127.0.0.1', self.get_http_port()))
        yield stream.write(b'\r\n'.join([b'POST /hello HTTP/1.1', b'Content-Length: 1024', b'Expect: 100-continue', b'Connection: close', b'\r\n']))
        data = (yield stream.read_until(b'\r\n\r\n'))
        self.assertTrue(data.startswith(b'HTTP/1.1 100 '), data)
        stream.write(b'a' * 1024)
        first_line = (yield stream.read_until(b'\r\n'))
        self.assertTrue(first_line.startswith(b'HTTP/1.1 200'), first_line)
        header_data = (yield stream.read_until(b'\r\n\r\n'))
        headers = HTTPHeaders.parse(native_str(header_data.decode('latin1')))
        body = (yield stream.read_bytes(int(headers['Content-Length'])))
        self.assertEqual(body, b'Got 1024 bytes in POST')
        stream.close()