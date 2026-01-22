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
class AsyncIOLoopTest(AsyncTestCase):

    @property
    def asyncio_loop(self):
        return self.io_loop.asyncio_loop

    def test_asyncio_callback(self):

        async def add_callback():
            asyncio.get_event_loop().call_soon(self.stop)
        self.asyncio_loop.run_until_complete(add_callback())
        self.wait()

    @gen_test
    def test_asyncio_future(self):
        x = (yield asyncio.ensure_future(asyncio.get_event_loop().run_in_executor(None, lambda: 42)))
        self.assertEqual(x, 42)

    @gen_test
    def test_asyncio_yield_from(self):

        @gen.coroutine
        def f():
            event_loop = asyncio.get_event_loop()
            x = (yield from event_loop.run_in_executor(None, lambda: 42))
            return x
        result = (yield f())
        self.assertEqual(result, 42)

    def test_asyncio_adapter(self):

        @gen.coroutine
        def tornado_coroutine():
            yield gen.moment
            raise gen.Return(42)

        async def native_coroutine_without_adapter():
            return await tornado_coroutine()

        async def native_coroutine_with_adapter():
            return await to_asyncio_future(tornado_coroutine())

        async def native_coroutine_with_adapter2():
            return await to_asyncio_future(native_coroutine_without_adapter())
        self.assertEqual(self.io_loop.run_sync(native_coroutine_without_adapter), 42)
        self.assertEqual(self.io_loop.run_sync(native_coroutine_with_adapter), 42)
        self.assertEqual(self.io_loop.run_sync(native_coroutine_with_adapter2), 42)
        self.assertEqual(self.asyncio_loop.run_until_complete(native_coroutine_without_adapter()), 42)
        self.assertEqual(self.asyncio_loop.run_until_complete(native_coroutine_with_adapter()), 42)
        self.assertEqual(self.asyncio_loop.run_until_complete(native_coroutine_with_adapter2()), 42)

    def test_add_thread_close_idempotent(self):
        loop = AddThreadSelectorEventLoop(asyncio.get_event_loop())
        loop.close()
        loop.close()