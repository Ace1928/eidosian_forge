import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
class LockTests(AsyncTestCase):

    def test_repr(self):
        lock = locks.Lock()
        repr(lock)
        lock.acquire()
        repr(lock)

    def test_acquire_release(self):
        lock = locks.Lock()
        self.assertTrue(asyncio.ensure_future(lock.acquire()).done())
        future = asyncio.ensure_future(lock.acquire())
        self.assertFalse(future.done())
        lock.release()
        self.assertTrue(future.done())

    @gen_test
    def test_acquire_fifo(self):
        lock = locks.Lock()
        self.assertTrue(asyncio.ensure_future(lock.acquire()).done())
        N = 5
        history = []

        @gen.coroutine
        def f(idx):
            with (yield lock.acquire()):
                history.append(idx)
        futures = [f(i) for i in range(N)]
        self.assertFalse(any((future.done() for future in futures)))
        lock.release()
        yield futures
        self.assertEqual(list(range(N)), history)

    @gen_test
    def test_acquire_fifo_async_with(self):
        lock = locks.Lock()
        self.assertTrue(asyncio.ensure_future(lock.acquire()).done())
        N = 5
        history = []

        async def f(idx):
            async with lock:
                history.append(idx)
        futures = [f(i) for i in range(N)]
        lock.release()
        yield futures
        self.assertEqual(list(range(N)), history)

    @gen_test
    def test_acquire_timeout(self):
        lock = locks.Lock()
        lock.acquire()
        with self.assertRaises(gen.TimeoutError):
            yield lock.acquire(timeout=timedelta(seconds=0.01))
        self.assertFalse(asyncio.ensure_future(lock.acquire()).done())

    def test_multi_release(self):
        lock = locks.Lock()
        self.assertRaises(RuntimeError, lock.release)
        lock.acquire()
        lock.release()
        self.assertRaises(RuntimeError, lock.release)

    @gen_test
    def test_yield_lock(self):
        with self.assertRaises(gen.BadYieldError):
            with (yield locks.Lock()):
                pass

    def test_context_manager_misuse(self):
        with self.assertRaises(RuntimeError):
            with locks.Lock():
                pass