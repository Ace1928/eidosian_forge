from tornado.concurrent import Future
from tornado import gen
from tornado.escape import (
from tornado.httpclient import HTTPClientError
from tornado.httputil import format_timestamp
from tornado.iostream import IOStream
from tornado import locale
from tornado.locks import Event
from tornado.log import app_log, gen_log
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.template import DictLoader
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, ExpectLog, gen_test
from tornado.test.util import ignore_deprecation
from tornado.util import ObjectDict, unicode_type
from tornado.web import (
import binascii
import contextlib
import copy
import datetime
import email.utils
import gzip
from io import BytesIO
import itertools
import logging
import os
import re
import socket
import typing  # noqa: F401
import unittest
import urllib.parse
class StreamingRequestBodyTest(WebTestCase):

    def get_handlers(self):

        @stream_request_body
        class StreamingBodyHandler(RequestHandler):

            def initialize(self, test):
                self.test = test

            def prepare(self):
                self.test.prepared.set_result(None)

            def data_received(self, data):
                self.test.data.set_result(data)

            def get(self):
                self.test.finished.set_result(None)
                self.write({})

        @stream_request_body
        class EarlyReturnHandler(RequestHandler):

            def prepare(self):
                raise HTTPError(401)

        @stream_request_body
        class CloseDetectionHandler(RequestHandler):

            def initialize(self, test):
                self.test = test

            def on_connection_close(self):
                super().on_connection_close()
                self.test.close_future.set_result(None)
        return [('/stream_body', StreamingBodyHandler, dict(test=self)), ('/early_return', EarlyReturnHandler), ('/close_detection', CloseDetectionHandler, dict(test=self))]

    def connect(self, url, connection_close):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        s.connect(('127.0.0.1', self.get_http_port()))
        stream = IOStream(s)
        stream.write(b'GET ' + url + b' HTTP/1.1\r\n')
        if connection_close:
            stream.write(b'Connection: close\r\n')
        stream.write(b'Transfer-Encoding: chunked\r\n\r\n')
        return stream

    @gen_test
    def test_streaming_body(self):
        self.prepared = Future()
        self.data = Future()
        self.finished = Future()
        stream = self.connect(b'/stream_body', connection_close=True)
        yield self.prepared
        stream.write(b'4\r\nasdf\r\n')
        data = (yield self.data)
        self.assertEqual(data, b'asdf')
        self.data = Future()
        stream.write(b'4\r\nqwer\r\n')
        data = (yield self.data)
        self.assertEqual(data, b'qwer')
        stream.write(b'0\r\n\r\n')
        yield self.finished
        data = (yield stream.read_until_close())
        self.assertTrue(data.endswith(b'{}'))
        stream.close()

    @gen_test
    def test_early_return(self):
        stream = self.connect(b'/early_return', connection_close=False)
        data = (yield stream.read_until_close())
        self.assertTrue(data.startswith(b'HTTP/1.1 401'))

    @gen_test
    def test_early_return_with_data(self):
        stream = self.connect(b'/early_return', connection_close=False)
        stream.write(b'4\r\nasdf\r\n')
        data = (yield stream.read_until_close())
        self.assertTrue(data.startswith(b'HTTP/1.1 401'))

    @gen_test
    def test_close_during_upload(self):
        self.close_future = Future()
        stream = self.connect(b'/close_detection', connection_close=False)
        stream.close()
        yield self.close_future