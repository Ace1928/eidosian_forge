import asyncio
from concurrent import futures
import gc
import datetime
import platform
import sys
import time
import weakref
import unittest
from tornado.concurrent import Future
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, ExpectLog, gen_test
from tornado.test.util import skipOnTravis, skipNotCPython
from tornado.web import Application, RequestHandler, HTTPError
from tornado import gen
import typing
def test_sync_raise_return_value_tuple(self):

    @gen.coroutine
    def f():
        raise gen.Return((1, 2))
    self.assertEqual((1, 2), self.io_loop.run_sync(f))