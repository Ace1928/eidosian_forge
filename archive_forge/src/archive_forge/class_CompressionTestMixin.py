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
class CompressionTestMixin(object):
    MESSAGE = 'Hello world. Testing 123 123'

    def get_app(self):

        class LimitedHandler(TestWebSocketHandler):

            @property
            def max_message_size(self):
                return 1024

            def on_message(self, message):
                self.write_message(str(len(message)))
        return Application([('/echo', EchoHandler, dict(compression_options=self.get_server_compression_options())), ('/limited', LimitedHandler, dict(compression_options=self.get_server_compression_options()))])

    def get_server_compression_options(self):
        return None

    def get_client_compression_options(self):
        return None

    def verify_wire_bytes(self, bytes_in: int, bytes_out: int) -> None:
        raise NotImplementedError()

    @gen_test
    def test_message_sizes(self: typing.Any):
        ws = (yield self.ws_connect('/echo', compression_options=self.get_client_compression_options()))
        for i in range(3):
            ws.write_message(self.MESSAGE)
            response = (yield ws.read_message())
            self.assertEqual(response, self.MESSAGE)
        self.assertEqual(ws.protocol._message_bytes_out, len(self.MESSAGE) * 3)
        self.assertEqual(ws.protocol._message_bytes_in, len(self.MESSAGE) * 3)
        self.verify_wire_bytes(ws.protocol._wire_bytes_in, ws.protocol._wire_bytes_out)

    @gen_test
    def test_size_limit(self: typing.Any):
        ws = (yield self.ws_connect('/limited', compression_options=self.get_client_compression_options()))
        ws.write_message('a' * 128)
        response = (yield ws.read_message())
        self.assertEqual(response, '128')
        ws.write_message('a' * 2048)
        response = (yield ws.read_message())
        self.assertIsNone(response)