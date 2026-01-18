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
def test_set_cookie_expires_days(self):
    response = self.fetch('/set_expires_days')
    header = response.headers.get('Set-Cookie')
    assert header is not None
    match = re.match('foo=bar; expires=(?P<expires>.+); Path=/', header)
    assert match is not None
    expires = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=10)
    header_expires = email.utils.parsedate_to_datetime(match.groupdict()['expires'])
    self.assertTrue(abs((expires - header_expires).total_seconds()) < 10)