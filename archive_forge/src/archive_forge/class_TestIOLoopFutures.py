import asyncio
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
from collections.abc import Generator
import contextlib
import datetime
import functools
import socket
import subprocess
import sys
import threading
import time
import types
from unittest import mock
import unittest
from tornado.escape import native_str
from tornado import gen
from tornado.ioloop import IOLoop, TimeoutError, PeriodicCallback
from tornado.log import app_log
from tornado.testing import (
from tornado.test.util import (
from tornado.concurrent import Future
import typing
class TestIOLoopFutures(AsyncTestCase):

    def test_add_future_threads(self):
        with futures.ThreadPoolExecutor(1) as pool:

            def dummy():
                pass
            self.io_loop.add_future(pool.submit(dummy), lambda future: self.stop(future))
            future = self.wait()
            self.assertTrue(future.done())
            self.assertTrue(future.result() is None)

    @gen_test
    def test_run_in_executor_gen(self):
        event1 = threading.Event()
        event2 = threading.Event()

        def sync_func(self_event, other_event):
            self_event.set()
            other_event.wait()
            return self_event
        res = (yield [IOLoop.current().run_in_executor(None, sync_func, event1, event2), IOLoop.current().run_in_executor(None, sync_func, event2, event1)])
        self.assertEqual([event1, event2], res)

    @gen_test
    def test_run_in_executor_native(self):
        event1 = threading.Event()
        event2 = threading.Event()

        def sync_func(self_event, other_event):
            self_event.set()
            other_event.wait()
            return self_event

        async def async_wrapper(self_event, other_event):
            return await IOLoop.current().run_in_executor(None, sync_func, self_event, other_event)
        res = (yield [async_wrapper(event1, event2), async_wrapper(event2, event1)])
        self.assertEqual([event1, event2], res)

    @gen_test
    def test_set_default_executor(self):
        count = [0]

        class MyExecutor(futures.ThreadPoolExecutor):

            def submit(self, func, *args):
                count[0] += 1
                return super().submit(func, *args)
        event = threading.Event()

        def sync_func():
            event.set()
        executor = MyExecutor(1)
        loop = IOLoop.current()
        loop.set_default_executor(executor)
        yield loop.run_in_executor(None, sync_func)
        self.assertEqual(1, count[0])
        self.assertTrue(event.is_set())