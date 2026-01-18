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
def test_websocket_callbacks(self):
    websocket_connect('ws://127.0.0.1:%d/echo' % self.get_http_port(), callback=self.stop)
    ws = self.wait().result()
    ws.write_message('hello')
    ws.read_message(self.stop)
    response = self.wait().result()
    self.assertEqual(response, 'hello')
    self.close_future.add_done_callback(lambda f: self.stop())
    ws.close()
    self.wait()