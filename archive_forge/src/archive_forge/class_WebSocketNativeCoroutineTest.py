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
class WebSocketNativeCoroutineTest(WebSocketBaseTestCase):

    def get_app(self):
        return Application([('/native', NativeCoroutineOnMessageHandler)])

    @gen_test
    def test_native_coroutine(self):
        ws = (yield self.ws_connect('/native'))
        yield ws.write_message('hello1')
        yield ws.write_message('hello2')
        res = (yield ws.read_message())
        self.assertEqual(res, 'hello1')
        res = (yield ws.read_message())
        self.assertEqual(res, 'hello2')