from tornado import gen, netutil
from tornado.escape import (
from tornado.http1connection import HTTP1Connection
from tornado.httpclient import HTTPError
from tornado.httpserver import HTTPServer
from tornado.httputil import (
from tornado.iostream import IOStream
from tornado.locks import Event
from tornado.log import gen_log, app_log
from tornado.netutil import ssl_options_to_context
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.testing import (
from tornado.test.util import skipOnTravis
from tornado.web import Application, RequestHandler, stream_request_body
from contextlib import closing
import datetime
import gzip
import logging
import os
import shutil
import socket
import ssl
import sys
import tempfile
import textwrap
import unittest
import urllib.parse
from io import BytesIO
import typing
def test_scheme_headers(self):
    self.assertEqual(self.fetch_json('/')['remote_protocol'], 'http')
    https_scheme = {'X-Scheme': 'https'}
    self.assertEqual(self.fetch_json('/', headers=https_scheme)['remote_protocol'], 'https')
    https_forwarded = {'X-Forwarded-Proto': 'https'}
    self.assertEqual(self.fetch_json('/', headers=https_forwarded)['remote_protocol'], 'https')
    https_multi_forwarded = {'X-Forwarded-Proto': 'https , http'}
    self.assertEqual(self.fetch_json('/', headers=https_multi_forwarded)['remote_protocol'], 'http')
    http_multi_forwarded = {'X-Forwarded-Proto': 'http,https'}
    self.assertEqual(self.fetch_json('/', headers=http_multi_forwarded)['remote_protocol'], 'https')
    bad_forwarded = {'X-Forwarded-Proto': 'unknown'}
    self.assertEqual(self.fetch_json('/', headers=bad_forwarded)['remote_protocol'], 'http')