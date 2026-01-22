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
class SSLTestMixin(object):

    def get_ssl_options(self):
        return dict(ssl_version=self.get_ssl_version(), **AsyncHTTPSTestCase.default_ssl_options())

    def get_ssl_version(self):
        raise NotImplementedError()

    def test_ssl(self: typing.Any):
        response = self.fetch('/')
        self.assertEqual(response.body, b'Hello world')

    def test_large_post(self: typing.Any):
        response = self.fetch('/', method='POST', body='A' * 5000)
        self.assertEqual(response.body, b'Got 5000 bytes in POST')

    def test_non_ssl_request(self: typing.Any):
        with ExpectLog(gen_log, '(SSL Error|uncaught exception)'):
            with ExpectLog(gen_log, 'Uncaught exception', required=False):
                with self.assertRaises((IOError, HTTPError)):
                    self.fetch(self.get_url('/').replace('https:', 'http:'), request_timeout=3600, connect_timeout=3600, raise_error=True)

    def test_error_logging(self: typing.Any):
        with ExpectLog(gen_log, 'SSL Error') as expect_log:
            with self.assertRaises((IOError, HTTPError)):
                self.fetch(self.get_url('/').replace('https:', 'http:'), raise_error=True)
        self.assertFalse(expect_log.logged_stack)