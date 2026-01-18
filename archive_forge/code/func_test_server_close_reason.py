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
def test_server_close_reason(self):
    ws = (yield self.ws_connect('/close_reason'))
    msg = (yield ws.read_message())
    self.assertIs(msg, None)
    self.assertEqual(ws.close_code, 1001)
    self.assertEqual(ws.close_reason, 'goodbye')
    code, reason = (yield self.close_future)
    self.assertEqual(code, 1001)