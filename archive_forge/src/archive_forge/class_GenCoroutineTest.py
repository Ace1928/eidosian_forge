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
class GenCoroutineTest(AsyncTestCase):

    def setUp(self):
        self.finished = False
        super().setUp()

    def tearDown(self):
        super().tearDown()
        assert self.finished

    def test_attributes(self):
        self.finished = True

        def f():
            yield gen.moment
        coro = gen.coroutine(f)
        self.assertEqual(coro.__name__, f.__name__)
        self.assertEqual(coro.__module__, f.__module__)
        self.assertIs(coro.__wrapped__, f)

    def test_is_coroutine_function(self):
        self.finished = True

        def f():
            yield gen.moment
        coro = gen.coroutine(f)
        self.assertFalse(gen.is_coroutine_function(f))
        self.assertTrue(gen.is_coroutine_function(coro))
        self.assertFalse(gen.is_coroutine_function(coro()))

    @gen_test
    def test_sync_gen_return(self):

        @gen.coroutine
        def f():
            raise gen.Return(42)
        result = (yield f())
        self.assertEqual(result, 42)
        self.finished = True

    @gen_test
    def test_async_gen_return(self):

        @gen.coroutine
        def f():
            yield gen.moment
            raise gen.Return(42)
        result = (yield f())
        self.assertEqual(result, 42)
        self.finished = True

    @gen_test
    def test_sync_return(self):

        @gen.coroutine
        def f():
            return 42
        result = (yield f())
        self.assertEqual(result, 42)
        self.finished = True

    @gen_test
    def test_async_return(self):

        @gen.coroutine
        def f():
            yield gen.moment
            return 42
        result = (yield f())
        self.assertEqual(result, 42)
        self.finished = True

    @gen_test
    def test_async_early_return(self):

        @gen.coroutine
        def f():
            if True:
                return 42
            yield gen.Task(self.io_loop.add_callback)
        result = (yield f())
        self.assertEqual(result, 42)
        self.finished = True

    @gen_test
    def test_async_await(self):

        @gen.coroutine
        def f1():
            yield gen.moment
            raise gen.Return(42)

        async def f2():
            result = await f1()
            return result
        result = (yield f2())
        self.assertEqual(result, 42)
        self.finished = True

    @gen_test
    def test_asyncio_sleep_zero(self):

        async def f():
            import asyncio
            await asyncio.sleep(0)
            return 42
        result = (yield f())
        self.assertEqual(result, 42)
        self.finished = True

    @gen_test
    def test_async_await_mixed_multi_native_future(self):

        @gen.coroutine
        def f1():
            yield gen.moment

        async def f2():
            await f1()
            return 42

        @gen.coroutine
        def f3():
            yield gen.moment
            raise gen.Return(43)
        results = (yield [f2(), f3()])
        self.assertEqual(results, [42, 43])
        self.finished = True

    @gen_test
    def test_async_with_timeout(self):

        async def f1():
            return 42
        result = (yield gen.with_timeout(datetime.timedelta(hours=1), f1()))
        self.assertEqual(result, 42)
        self.finished = True

    @gen_test
    def test_sync_return_no_value(self):

        @gen.coroutine
        def f():
            return
        result = (yield f())
        self.assertEqual(result, None)
        self.finished = True

    @gen_test
    def test_async_return_no_value(self):

        @gen.coroutine
        def f():
            yield gen.moment
            return
        result = (yield f())
        self.assertEqual(result, None)
        self.finished = True

    @gen_test
    def test_sync_raise(self):

        @gen.coroutine
        def f():
            1 / 0
        future = f()
        with self.assertRaises(ZeroDivisionError):
            yield future
        self.finished = True

    @gen_test
    def test_async_raise(self):

        @gen.coroutine
        def f():
            yield gen.moment
            1 / 0
        future = f()
        with self.assertRaises(ZeroDivisionError):
            yield future
        self.finished = True

    @gen_test
    def test_replace_yieldpoint_exception(self):

        @gen.coroutine
        def f1():
            1 / 0

        @gen.coroutine
        def f2():
            try:
                yield f1()
            except ZeroDivisionError:
                raise KeyError()
        future = f2()
        with self.assertRaises(KeyError):
            yield future
        self.finished = True

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

    @gen_test
    def test_moment(self):
        calls = []

        @gen.coroutine
        def f(name, yieldable):
            for i in range(5):
                calls.append(name)
                yield yieldable
        immediate = Future()
        immediate.set_result(None)
        yield [f('a', immediate), f('b', immediate)]
        self.assertEqual(''.join(calls), 'aaaaabbbbb')
        calls = []
        yield [f('a', gen.moment), f('b', gen.moment)]
        self.assertEqual(''.join(calls), 'ababababab')
        self.finished = True
        calls = []
        yield [f('a', gen.moment), f('b', immediate)]
        self.assertEqual(''.join(calls), 'abbbbbaaaa')

    @gen_test
    def test_sleep(self):
        yield gen.sleep(0.01)
        self.finished = True

    @gen_test
    def test_py3_leak_exception_context(self):

        class LeakedException(Exception):
            pass

        @gen.coroutine
        def inner(iteration):
            raise LeakedException(iteration)
        try:
            yield inner(1)
        except LeakedException as e:
            self.assertEqual(str(e), '1')
            self.assertIsNone(e.__context__)
        try:
            yield inner(2)
        except LeakedException as e:
            self.assertEqual(str(e), '2')
            self.assertIsNone(e.__context__)
        self.finished = True

    @skipNotCPython
    @unittest.skipIf((3,) < sys.version_info < (3, 6), 'asyncio.Future has reference cycles')
    def test_coroutine_refcounting(self):

        @gen.coroutine
        def inner():

            class Foo(object):
                pass
            local_var = Foo()
            self.local_ref = weakref.ref(local_var)

            def dummy():
                pass
            yield gen.coroutine(dummy)()
            raise ValueError('Some error')

        @gen.coroutine
        def inner2():
            try:
                yield inner()
            except ValueError:
                pass
        self.io_loop.run_sync(inner2, timeout=3)
        self.assertIs(self.local_ref(), None)
        self.finished = True

    def test_asyncio_future_debug_info(self):
        self.finished = True
        asyncio_loop = asyncio.get_event_loop()
        self.addCleanup(asyncio_loop.set_debug, asyncio_loop.get_debug())
        asyncio_loop.set_debug(True)

        def f():
            yield gen.moment
        coro = gen.coroutine(f)()
        self.assertIsInstance(coro, asyncio.Future)
        expected = 'created at %s:%d' % (__file__, f.__code__.co_firstlineno + 3)
        actual = repr(coro)
        self.assertIn(expected, actual)

    @gen_test
    def test_asyncio_gather(self):

        @gen.coroutine
        def f():
            yield gen.moment
            raise gen.Return(1)
        ret = (yield asyncio.gather(f(), f()))
        self.assertEqual(ret, [1, 1])
        self.finished = True