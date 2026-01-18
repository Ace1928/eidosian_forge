import asyncio
import contextlib
import functools
import socket
import traceback
import typing
import unittest
from tornado.concurrent import Future
from tornado import gen
from tornado.httpclient import HTTPError, HTTPRequest
from tornado.locks import Event
from tornado.log import gen_log, app_log
from tornado.netutil import Resolver
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.template import DictLoader
from tornado.testing import AsyncHTTPTestCase, gen_test, bind_unused_port, ExpectLog
from tornado.web import Application, RequestHandler
from tornado.websocket import (
@gen_test
def test_websocket_headers(self):
    with contextlib.closing((yield websocket_connect(HTTPRequest('ws://127.0.0.1:%d/header' % self.get_http_port(), headers={'X-Test': 'hello'})))) as ws:
        response = (yield ws.read_message())
        self.assertEqual(response, 'hello')