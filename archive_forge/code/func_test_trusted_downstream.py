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
def test_trusted_downstream(self):
    valid_ipv4_list = {'X-Forwarded-For': '127.0.0.1, 4.4.4.4, 5.5.5.5'}
    resp = self.fetch('/', headers=valid_ipv4_list)
    if resp.headers['request-version'].startswith('HTTP/2'):
        self.skipTest('requires HTTP/1.x')
    result = json_decode(resp.body)
    self.assertEqual(result['remote_ip'], '4.4.4.4')