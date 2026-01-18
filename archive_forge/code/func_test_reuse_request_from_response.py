import base64
import binascii
from contextlib import closing
import copy
import gzip
import threading
import datetime
from io import BytesIO
import subprocess
import sys
import time
import typing  # noqa: F401
import unicodedata
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado import gen
from tornado.httpclient import (
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.iostream import IOStream
from tornado.log import gen_log, app_log
from tornado import netutil
from tornado.testing import AsyncHTTPTestCase, bind_unused_port, gen_test, ExpectLog
from tornado.test.util import skipOnTravis, ignore_deprecation
from tornado.web import Application, RequestHandler, url
from tornado.httputil import format_timestamp, HTTPHeaders
@gen_test
def test_reuse_request_from_response(self):
    url = self.get_url('/hello')
    response = (yield self.http_client.fetch(url))
    self.assertEqual(response.request.url, url)
    self.assertTrue(isinstance(response.request, HTTPRequest))
    response2 = (yield self.http_client.fetch(response.request))
    self.assertEqual(response2.body, b'Hello world!')