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
def test_see_other_redirect(self: typing.Any):
    for code in (302, 303):
        response = self.fetch('/see_other_post', method='POST', body='%d' % code)
        self.assertEqual(200, response.code)
        self.assertTrue(response.request.url.endswith('/see_other_post'))
        self.assertTrue(response.effective_url.endswith('/see_other_get'))
        self.assertEqual('POST', response.request.method)