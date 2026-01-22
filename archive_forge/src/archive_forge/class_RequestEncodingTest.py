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
class RequestEncodingTest(WebTestCase):

    def get_handlers(self):
        return [('/group/(.*)', EchoHandler), ('/slashes/([^/]*)/([^/]*)', EchoHandler)]

    def fetch_json(self, path):
        return json_decode(self.fetch(path).body)

    def test_group_question_mark(self):
        self.assertEqual(self.fetch_json('/group/%3F'), dict(path='/group/%3F', path_args=['?'], args={}))
        self.assertEqual(self.fetch_json('/group/%3F?%3F=%3F'), dict(path='/group/%3F', path_args=['?'], args={'?': ['?']}))

    def test_group_encoding(self):
        self.assertEqual(self.fetch_json('/group/%C3%A9?arg=%C3%A9'), {'path': '/group/%C3%A9', 'path_args': ['é'], 'args': {'arg': ['é']}})

    def test_slashes(self):
        self.assertEqual(self.fetch_json('/slashes/foo/bar'), dict(path='/slashes/foo/bar', path_args=['foo', 'bar'], args={}))
        self.assertEqual(self.fetch_json('/slashes/a%2Fb/c%2Fd'), dict(path='/slashes/a%2Fb/c%2Fd', path_args=['a/b', 'c/d'], args={}))

    def test_error(self):
        with ExpectLog(gen_log, '.*Invalid unicode'):
            self.fetch('/group/?arg=%25%e9')