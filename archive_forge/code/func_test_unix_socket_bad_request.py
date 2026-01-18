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
@gen_test
def test_unix_socket_bad_request(self):
    with ExpectLog(gen_log, 'Malformed HTTP message from', level=logging.INFO):
        self.stream.write(b'garbage\r\n\r\n')
        response = (yield self.stream.read_until_close())
    self.assertEqual(response, b'HTTP/1.1 400 Bad Request\r\n\r\n')