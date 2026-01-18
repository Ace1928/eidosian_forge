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
def test_static_range_if_none_match(self):
    response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=1-4', 'If-None-Match': b'"' + self.robots_txt_hash + b'"'})
    self.assertEqual(response.code, 304)
    self.assertEqual(response.body, b'')
    self.assertTrue('Content-Length' not in response.headers)
    self.assertEqual(utf8(response.headers['Etag']), b'"' + self.robots_txt_hash + b'"')