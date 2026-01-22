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
class URLSpecReverseTest(unittest.TestCase):

    def test_reverse(self):
        self.assertEqual('/favicon.ico', url('/favicon\\.ico', None).reverse())
        self.assertEqual('/favicon.ico', url('^/favicon\\.ico$', None).reverse())

    def test_non_reversible(self):
        paths = ['^/api/v\\d+/foo/(\\w+)$']
        for path in paths:
            url_spec = url(path, None)
            try:
                result = url_spec.reverse()
                self.fail('did not get expected exception when reversing %s. result: %s' % (path, result))
            except ValueError:
                pass

    def test_reverse_arguments(self):
        self.assertEqual('/api/v1/foo/bar', url('^/api/v1/foo/(\\w+)$', None).reverse('bar'))
        self.assertEqual('/api.v1/foo/5/icon.png', url('/api\\.v1/foo/([0-9]+)/icon\\.png', None).reverse(5))