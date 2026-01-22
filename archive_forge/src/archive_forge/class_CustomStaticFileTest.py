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
class CustomStaticFileTest(WebTestCase):

    def get_handlers(self):

        class MyStaticFileHandler(StaticFileHandler):

            @classmethod
            def make_static_url(cls, settings, path):
                version_hash = cls.get_version(settings, path)
                extension_index = path.rindex('.')
                before_version = path[:extension_index]
                after_version = path[extension_index + 1:]
                return '/static/%s.%s.%s' % (before_version, version_hash, after_version)

            def parse_url_path(self, url_path):
                extension_index = url_path.rindex('.')
                version_index = url_path.rindex('.', 0, extension_index)
                return '%s%s' % (url_path[:version_index], url_path[extension_index:])

            @classmethod
            def get_absolute_path(cls, settings, path):
                return 'CustomStaticFileTest:' + path

            def validate_absolute_path(self, root, absolute_path):
                return absolute_path

            @classmethod
            def get_content(self, path, start=None, end=None):
                assert start is None and end is None
                if path == 'CustomStaticFileTest:foo.txt':
                    return b'bar'
                raise Exception('unexpected path %r' % path)

            def get_content_size(self):
                if self.absolute_path == 'CustomStaticFileTest:foo.txt':
                    return 3
                raise Exception('unexpected path %r' % self.absolute_path)

            def get_modified_time(self):
                return None

            @classmethod
            def get_version(cls, settings, path):
                return '42'

        class StaticUrlHandler(RequestHandler):

            def get(self, path):
                self.write(self.static_url(path))
        self.static_handler_class = MyStaticFileHandler
        return [('/static_url/(.*)', StaticUrlHandler)]

    def get_app_kwargs(self):
        return dict(static_path='dummy', static_handler_class=self.static_handler_class)

    def test_serve(self):
        response = self.fetch('/static/foo.42.txt')
        self.assertEqual(response.body, b'bar')

    def test_static_url(self):
        with ExpectLog(gen_log, 'Could not open static file', required=False):
            response = self.fetch('/static_url/foo.txt')
            self.assertEqual(response.body, b'/static/foo.42.txt')