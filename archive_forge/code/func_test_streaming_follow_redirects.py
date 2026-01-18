import collections
from contextlib import closing
import errno
import logging
import os
import re
import socket
import ssl
import sys
import typing  # noqa: F401
from tornado.escape import to_unicode, utf8
from tornado import gen, version
from tornado.httpclient import AsyncHTTPClient
from tornado.httputil import HTTPHeaders, ResponseStartLine
from tornado.ioloop import IOLoop
from tornado.iostream import UnsatisfiableReadError
from tornado.locks import Event
from tornado.log import gen_log
from tornado.netutil import Resolver, bind_sockets
from tornado.simple_httpclient import (
from tornado.test.httpclient_test import (
from tornado.test import httpclient_test
from tornado.testing import (
from tornado.test.util import skipOnTravis, skipIfNoIPv6, refusing_port
from tornado.web import RequestHandler, Application, url, stream_request_body
def test_streaming_follow_redirects(self: typing.Any):
    headers = []
    chunk_bytes = []
    self.fetch('/redirect?url=/hello', header_callback=headers.append, streaming_callback=chunk_bytes.append)
    chunks = list(map(to_unicode, chunk_bytes))
    self.assertEqual(chunks, ['Hello world!'])
    num_start_lines = len([h for h in headers if h.startswith('HTTP/')])
    self.assertEqual(num_start_lines, 1)