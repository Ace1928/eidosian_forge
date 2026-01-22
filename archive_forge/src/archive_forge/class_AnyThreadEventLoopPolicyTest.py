import asyncio
import threading
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from tornado import gen
from tornado.ioloop import IOLoop
from tornado.platform.asyncio import (
from tornado.testing import AsyncTestCase, gen_test
class AnyThreadEventLoopPolicyTest(unittest.TestCase):

    def setUp(self):
        self.orig_policy = asyncio.get_event_loop_policy()
        self.executor = ThreadPoolExecutor(1)

    def tearDown(self):
        asyncio.set_event_loop_policy(self.orig_policy)
        self.executor.shutdown()

    def get_event_loop_on_thread(self):

        def get_and_close_event_loop():
            """Get the event loop. Close it if one is returned.

            Returns the (closed) event loop. This is a silly thing
            to do and leaves the thread in a broken state, but it's
            enough for this test. Closing the loop avoids resource
            leak warnings.
            """
            loop = asyncio.get_event_loop()
            loop.close()
            return loop
        future = self.executor.submit(get_and_close_event_loop)
        return future.result()

    def test_asyncio_accessor(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            self.assertRaises(RuntimeError, self.executor.submit(asyncio.get_event_loop).result)
            asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
            self.assertIsInstance(self.executor.submit(asyncio.get_event_loop).result(), asyncio.AbstractEventLoop)
            self.executor.submit(lambda: asyncio.get_event_loop().close()).result()

    def test_tornado_accessor(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            self.assertIsInstance(self.executor.submit(IOLoop.current).result(), IOLoop)
            self.executor.submit(lambda: asyncio.get_event_loop().close()).result()
            asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
            self.assertIsInstance(self.executor.submit(IOLoop.current).result(), IOLoop)
            self.executor.submit(lambda: asyncio.get_event_loop().close()).result()