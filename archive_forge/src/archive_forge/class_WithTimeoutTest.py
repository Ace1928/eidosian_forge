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
class WithTimeoutTest(AsyncTestCase):

    @gen_test
    def test_timeout(self):
        with self.assertRaises(gen.TimeoutError):
            yield gen.with_timeout(datetime.timedelta(seconds=0.1), Future())

    @gen_test
    def test_completes_before_timeout(self):
        future = Future()
        self.io_loop.add_timeout(datetime.timedelta(seconds=0.1), lambda: future.set_result('asdf'))
        result = (yield gen.with_timeout(datetime.timedelta(seconds=3600), future))
        self.assertEqual(result, 'asdf')

    @gen_test
    def test_fails_before_timeout(self):
        future = Future()
        self.io_loop.add_timeout(datetime.timedelta(seconds=0.1), lambda: future.set_exception(ZeroDivisionError()))
        with self.assertRaises(ZeroDivisionError):
            yield gen.with_timeout(datetime.timedelta(seconds=3600), future)

    @gen_test
    def test_already_resolved(self):
        future = Future()
        future.set_result('asdf')
        result = (yield gen.with_timeout(datetime.timedelta(seconds=3600), future))
        self.assertEqual(result, 'asdf')

    @gen_test
    def test_timeout_concurrent_future(self):
        with futures.ThreadPoolExecutor(1) as executor:
            with self.assertRaises(gen.TimeoutError):
                yield gen.with_timeout(self.io_loop.time(), executor.submit(time.sleep, 0.1))

    @gen_test
    def test_completed_concurrent_future(self):
        with futures.ThreadPoolExecutor(1) as executor:

            def dummy():
                pass
            f = executor.submit(dummy)
            f.result()
            yield gen.with_timeout(datetime.timedelta(seconds=3600), f)

    @gen_test
    def test_normal_concurrent_future(self):
        with futures.ThreadPoolExecutor(1) as executor:
            yield gen.with_timeout(datetime.timedelta(seconds=3600), executor.submit(lambda: time.sleep(0.01)))