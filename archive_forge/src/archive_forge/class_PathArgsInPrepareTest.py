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
class PathArgsInPrepareTest(WebTestCase):

    class Handler(RequestHandler):

        def prepare(self):
            self.write(dict(args=self.path_args, kwargs=self.path_kwargs))

        def get(self, path):
            assert path == 'foo'
            self.finish()

    def get_handlers(self):
        return [('/pos/(.*)', self.Handler), ('/kw/(?P<path>.*)', self.Handler)]

    def test_pos(self):
        response = self.fetch('/pos/foo')
        response.rethrow()
        data = json_decode(response.body)
        self.assertEqual(data, {'args': ['foo'], 'kwargs': {}})

    def test_kw(self):
        response = self.fetch('/kw/foo')
        response.rethrow()
        data = json_decode(response.body)
        self.assertEqual(data, {'args': [], 'kwargs': {'path': 'foo'}})