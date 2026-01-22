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
class AllHTTPMethodsTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def method(self):
            assert self.request.method is not None
            self.write(self.request.method)
        get = delete = options = post = put = method

    def test_standard_methods(self):
        response = self.fetch('/', method='HEAD')
        self.assertEqual(response.body, b'')
        for method in ['GET', 'DELETE', 'OPTIONS']:
            response = self.fetch('/', method=method)
            self.assertEqual(response.body, utf8(method))
        for method in ['POST', 'PUT']:
            response = self.fetch('/', method=method, body=b'')
            self.assertEqual(response.body, utf8(method))