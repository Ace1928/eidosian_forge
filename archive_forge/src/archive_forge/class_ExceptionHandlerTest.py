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
class ExceptionHandlerTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            exc = self.get_argument('exc')
            if exc == 'http':
                raise HTTPError(410, 'no longer here')
            elif exc == 'zero':
                1 / 0
            elif exc == 'permission':
                raise PermissionError('not allowed')

        def write_error(self, status_code, **kwargs):
            if 'exc_info' in kwargs:
                typ, value, tb = kwargs['exc_info']
                if isinstance(value, PermissionError):
                    self.set_status(403)
                    self.write('PermissionError')
                    return
            RequestHandler.write_error(self, status_code, **kwargs)

        def log_exception(self, typ, value, tb):
            if isinstance(value, PermissionError):
                app_log.warning('custom logging for PermissionError: %s', value.args[0])
            else:
                RequestHandler.log_exception(self, typ, value, tb)

    def test_http_error(self):
        with ExpectLog(gen_log, '.*no longer here'):
            response = self.fetch('/?exc=http')
            self.assertEqual(response.code, 410)

    def test_unknown_error(self):
        with ExpectLog(app_log, 'Uncaught exception'):
            response = self.fetch('/?exc=zero')
            self.assertEqual(response.code, 500)

    def test_known_error(self):
        with ExpectLog(app_log, 'custom logging for PermissionError: not allowed'):
            response = self.fetch('/?exc=permission')
            self.assertEqual(response.code, 403)