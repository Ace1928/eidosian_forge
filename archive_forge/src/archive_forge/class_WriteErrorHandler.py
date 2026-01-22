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
class WriteErrorHandler(RequestHandler):

    def get(self):
        if self.get_argument('status', None):
            self.send_error(int(self.get_argument('status')))
        else:
            1 / 0

    def write_error(self, status_code, **kwargs):
        self.set_header('Content-Type', 'text/plain')
        if 'exc_info' in kwargs:
            self.write('Exception: %s' % kwargs['exc_info'][0].__name__)
        else:
            self.write('Status: %d' % status_code)