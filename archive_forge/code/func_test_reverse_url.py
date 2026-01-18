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
def test_reverse_url(self):
    self.assertEqual(self.app.reverse_url('decode_arg', 'foo'), '/decode_arg/foo')
    self.assertEqual(self.app.reverse_url('decode_arg', 42), '/decode_arg/42')
    self.assertEqual(self.app.reverse_url('decode_arg', b'\xe9'), '/decode_arg/%E9')
    self.assertEqual(self.app.reverse_url('decode_arg', 'é'), '/decode_arg/%C3%A9')
    self.assertEqual(self.app.reverse_url('decode_arg', '1 + 1'), '/decode_arg/1%20%2B%201')