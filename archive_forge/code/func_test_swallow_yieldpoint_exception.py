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
@gen_test
def test_swallow_yieldpoint_exception(self):

    @gen.coroutine
    def f1():
        1 / 0

    @gen.coroutine
    def f2():
        try:
            yield f1()
        except ZeroDivisionError:
            raise gen.Return(42)
    result = (yield f2())
    self.assertEqual(result, 42)
    self.finished = True