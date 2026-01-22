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
class IncorrectContentLengthTest(SimpleHandlerTestCase):

    def get_handlers(self):
        test = self
        self.server_error = None

        class TooHigh(RequestHandler):

            def get(self):
                self.set_header('Content-Length', '42')
                try:
                    self.finish('ok')
                except Exception as e:
                    test.server_error = e
                    raise

        class TooLow(RequestHandler):

            def get(self):
                self.set_header('Content-Length', '2')
                try:
                    self.finish('hello')
                except Exception as e:
                    test.server_error = e
                    raise
        return [('/high', TooHigh), ('/low', TooLow)]

    def test_content_length_too_high(self):
        with ExpectLog(app_log, '(Uncaught exception|Exception in callback)'):
            with ExpectLog(gen_log, '(Cannot send error response after headers written|Failed to flush partial response)'):
                with self.assertRaises(HTTPClientError):
                    self.fetch('/high', raise_error=True)
        self.assertEqual(str(self.server_error), 'Tried to write 40 bytes less than Content-Length')

    def test_content_length_too_low(self):
        with ExpectLog(app_log, '(Uncaught exception|Exception in callback)'):
            with ExpectLog(gen_log, '(Cannot send error response after headers written|Failed to flush partial response)'):
                with self.assertRaises(HTTPClientError):
                    self.fetch('/low', raise_error=True)
        self.assertEqual(str(self.server_error), 'Tried to write more data than Content-Length')