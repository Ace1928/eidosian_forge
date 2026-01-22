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
class MaskFunctionMixin(object):

    def mask(self, mask: bytes, data: bytes) -> bytes:
        raise NotImplementedError()

    def test_mask(self: typing.Any):
        self.assertEqual(self.mask(b'abcd', b''), b'')
        self.assertEqual(self.mask(b'abcd', b'b'), b'\x03')
        self.assertEqual(self.mask(b'abcd', b'54321'), b'TVPVP')
        self.assertEqual(self.mask(b'ZXCV', b'98765432'), b'c`t`olpd')
        self.assertEqual(self.mask(b'\x00\x01\x02\x03', b'\xff\xfb\xfd\xfc\xfe\xfa'), b'\xff\xfa\xff\xff\xfe\xfb')
        self.assertEqual(self.mask(b'\xff\xfb\xfd\xfc', b'\x00\x01\x02\x03\x04\x05'), b'\xff\xfa\xff\xff\xfb\xfe')