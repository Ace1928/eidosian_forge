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
def test_request_without_xprotocol(self):
    self.assertEqual(self.fetch_json('/')['remote_protocol'], 'https')
    http_scheme = {'X-Scheme': 'http'}
    self.assertEqual(self.fetch_json('/', headers=http_scheme)['remote_protocol'], 'http')
    bad_scheme = {'X-Scheme': 'unknown'}
    self.assertEqual(self.fetch_json('/', headers=bad_scheme)['remote_protocol'], 'https')