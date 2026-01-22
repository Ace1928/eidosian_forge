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
class SecureCookieV2Test(unittest.TestCase):
    KEY_VERSIONS = {0: 'ajklasdf0ojaisdf', 1: 'aslkjasaolwkjsdf'}

    def test_round_trip(self):
        handler = CookieTestRequestHandler()
        handler.set_signed_cookie('foo', b'bar', version=2)
        self.assertEqual(handler.get_signed_cookie('foo', min_version=2), b'bar')

    def test_key_version_roundtrip(self):
        handler = CookieTestRequestHandler(cookie_secret=self.KEY_VERSIONS, key_version=0)
        handler.set_signed_cookie('foo', b'bar')
        self.assertEqual(handler.get_signed_cookie('foo'), b'bar')

    def test_key_version_roundtrip_differing_version(self):
        handler = CookieTestRequestHandler(cookie_secret=self.KEY_VERSIONS, key_version=1)
        handler.set_signed_cookie('foo', b'bar')
        self.assertEqual(handler.get_signed_cookie('foo'), b'bar')

    def test_key_version_increment_version(self):
        handler = CookieTestRequestHandler(cookie_secret=self.KEY_VERSIONS, key_version=0)
        handler.set_signed_cookie('foo', b'bar')
        new_handler = CookieTestRequestHandler(cookie_secret=self.KEY_VERSIONS, key_version=1)
        new_handler._cookies = handler._cookies
        self.assertEqual(new_handler.get_signed_cookie('foo'), b'bar')

    def test_key_version_invalidate_version(self):
        handler = CookieTestRequestHandler(cookie_secret=self.KEY_VERSIONS, key_version=0)
        handler.set_signed_cookie('foo', b'bar')
        new_key_versions = self.KEY_VERSIONS.copy()
        new_key_versions.pop(0)
        new_handler = CookieTestRequestHandler(cookie_secret=new_key_versions, key_version=1)
        new_handler._cookies = handler._cookies
        self.assertEqual(new_handler.get_signed_cookie('foo'), None)