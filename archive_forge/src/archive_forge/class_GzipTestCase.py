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
class GzipTestCase(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            for v in self.get_arguments('vary'):
                self.add_header('Vary', v)
            self.write('hello world' + '!' * GZipContentEncoding.MIN_LENGTH)

    def get_app_kwargs(self):
        return dict(gzip=True, static_path=os.path.join(os.path.dirname(__file__), 'static'))

    def assert_compressed(self, response):
        self.assertEqual(response.headers.get('Content-Encoding', response.headers.get('X-Consumed-Content-Encoding')), 'gzip')

    def test_gzip(self):
        response = self.fetch('/')
        self.assert_compressed(response)
        self.assertEqual(response.headers['Vary'], 'Accept-Encoding')

    def test_gzip_static(self):
        response = self.fetch('/robots.txt')
        self.assert_compressed(response)
        self.assertEqual(response.headers['Vary'], 'Accept-Encoding')

    def test_gzip_not_requested(self):
        response = self.fetch('/', use_gzip=False)
        self.assertNotIn('Content-Encoding', response.headers)
        self.assertEqual(response.headers['Vary'], 'Accept-Encoding')

    def test_vary_already_present(self):
        response = self.fetch('/?vary=Accept-Language')
        self.assert_compressed(response)
        self.assertEqual([s.strip() for s in response.headers['Vary'].split(',')], ['Accept-Language', 'Accept-Encoding'])

    def test_vary_already_present_multiple(self):
        response = self.fetch('/?vary=Accept-Language&vary=Cookie')
        self.assert_compressed(response)
        self.assertEqual([s.strip() for s in response.headers['Vary'].split(',')], ['Accept-Language', 'Cookie', 'Accept-Encoding'])