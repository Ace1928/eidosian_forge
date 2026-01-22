import asyncio
from collections.abc import Generator
import functools
import inspect
import logging
import os
import re
import signal
import socket
import sys
import unittest
import warnings
from tornado import gen
from tornado.httpclient import AsyncHTTPClient, HTTPResponse
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop, TimeoutError
from tornado import netutil
from tornado.platform.asyncio import AsyncIOMainLoop
from tornado.process import Subprocess
from tornado.log import app_log
from tornado.util import raise_exc_info, basestring_type
from tornado.web import Application
import typing
from typing import Tuple, Any, Callable, Type, Dict, Union, Optional, Coroutine
from types import TracebackType
class AsyncTestCase(unittest.TestCase):
    """`~unittest.TestCase` subclass for testing `.IOLoop`-based
    asynchronous code.

    The unittest framework is synchronous, so the test must be
    complete by the time the test method returns. This means that
    asynchronous code cannot be used in quite the same way as usual
    and must be adapted to fit. To write your tests with coroutines,
    decorate your test methods with `tornado.testing.gen_test` instead
    of `tornado.gen.coroutine`.

    This class also provides the (deprecated) `stop()` and `wait()`
    methods for a more manual style of testing. The test method itself
    must call ``self.wait()``, and asynchronous callbacks should call
    ``self.stop()`` to signal completion.

    By default, a new `.IOLoop` is constructed for each test and is available
    as ``self.io_loop``.  If the code being tested requires a
    reused global `.IOLoop`, subclasses should override `get_new_ioloop` to return it,
    although this is deprecated as of Tornado 6.3.

    The `.IOLoop`'s ``start`` and ``stop`` methods should not be
    called directly.  Instead, use `self.stop <stop>` and `self.wait
    <wait>`.  Arguments passed to ``self.stop`` are returned from
    ``self.wait``.  It is possible to have multiple ``wait``/``stop``
    cycles in the same test.

    Example::

        # This test uses coroutine style.
        class MyTestCase(AsyncTestCase):
            @tornado.testing.gen_test
            def test_http_fetch(self):
                client = AsyncHTTPClient()
                response = yield client.fetch("http://www.tornadoweb.org")
                # Test contents of response
                self.assertIn("FriendFeed", response.body)

        # This test uses argument passing between self.stop and self.wait.
        class MyTestCase2(AsyncTestCase):
            def test_http_fetch(self):
                client = AsyncHTTPClient()
                client.fetch("http://www.tornadoweb.org/", self.stop)
                response = self.wait()
                # Test contents of response
                self.assertIn("FriendFeed", response.body)
    """

    def __init__(self, methodName: str='runTest') -> None:
        super().__init__(methodName)
        self.__stopped = False
        self.__running = False
        self.__failure = None
        self.__stop_args = None
        self.__timeout = None
        setattr(self, methodName, _TestMethodWrapper(getattr(self, methodName)))
        self._test_generator = None

    def setUp(self) -> None:
        py_ver = sys.version_info
        if (3, 10, 0) <= py_ver < (3, 10, 9) or (3, 11, 0) <= py_ver <= (3, 11, 1):
            setup_with_context_manager(self, warnings.catch_warnings())
            warnings.filterwarnings('ignore', message='There is no current event loop', category=DeprecationWarning, module='tornado\\..*')
        super().setUp()
        if type(self).get_new_ioloop is not AsyncTestCase.get_new_ioloop:
            warnings.warn('get_new_ioloop is deprecated', DeprecationWarning)
        self.io_loop = self.get_new_ioloop()
        asyncio.set_event_loop(self.io_loop.asyncio_loop)

    def tearDown(self) -> None:
        asyncio_loop = self.io_loop.asyncio_loop
        tasks = asyncio.all_tasks(asyncio_loop)
        tasks = [t for t in tasks if not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            done, pending = self.io_loop.run_sync(lambda: asyncio.wait(tasks))
            assert not pending
            for f in done:
                try:
                    f.result()
                except asyncio.CancelledError:
                    pass
        Subprocess.uninitialize()
        asyncio.set_event_loop(None)
        if not isinstance(self.io_loop, _NON_OWNED_IOLOOPS):
            self.io_loop.close(all_fds=True)
        super().tearDown()
        self.__rethrow()

    def get_new_ioloop(self) -> IOLoop:
        """Returns the `.IOLoop` to use for this test.

        By default, a new `.IOLoop` is created for each test.
        Subclasses may override this method to return
        `.IOLoop.current()` if it is not appropriate to use a new
        `.IOLoop` in each tests (for example, if there are global
        singletons using the default `.IOLoop`) or if a per-test event
        loop is being provided by another system (such as
        ``pytest-asyncio``).

        .. deprecated:: 6.3
           This method will be removed in Tornado 7.0.
        """
        return IOLoop(make_current=False)

    def _handle_exception(self, typ: Type[Exception], value: Exception, tb: TracebackType) -> bool:
        if self.__failure is None:
            self.__failure = (typ, value, tb)
        else:
            app_log.error('multiple unhandled exceptions in test', exc_info=(typ, value, tb))
        self.stop()
        return True

    def __rethrow(self) -> None:
        if self.__failure is not None:
            failure = self.__failure
            self.__failure = None
            raise_exc_info(failure)

    def run(self, result: Optional[unittest.TestResult]=None) -> Optional[unittest.TestResult]:
        ret = super().run(result)
        self.__rethrow()
        return ret

    def stop(self, _arg: Any=None, **kwargs: Any) -> None:
        """Stops the `.IOLoop`, causing one pending (or future) call to `wait()`
        to return.

        Keyword arguments or a single positional argument passed to `stop()` are
        saved and will be returned by `wait()`.

        .. deprecated:: 5.1

           `stop` and `wait` are deprecated; use ``@gen_test`` instead.
        """
        assert _arg is None or not kwargs
        self.__stop_args = kwargs or _arg
        if self.__running:
            self.io_loop.stop()
            self.__running = False
        self.__stopped = True

    def wait(self, condition: Optional[Callable[..., bool]]=None, timeout: Optional[float]=None) -> Any:
        """Runs the `.IOLoop` until stop is called or timeout has passed.

        In the event of a timeout, an exception will be thrown. The
        default timeout is 5 seconds; it may be overridden with a
        ``timeout`` keyword argument or globally with the
        ``ASYNC_TEST_TIMEOUT`` environment variable.

        If ``condition`` is not ``None``, the `.IOLoop` will be restarted
        after `stop()` until ``condition()`` returns ``True``.

        .. versionchanged:: 3.1
           Added the ``ASYNC_TEST_TIMEOUT`` environment variable.

        .. deprecated:: 5.1

           `stop` and `wait` are deprecated; use ``@gen_test`` instead.
        """
        if timeout is None:
            timeout = get_async_test_timeout()
        if not self.__stopped:
            if timeout:

                def timeout_func() -> None:
                    try:
                        raise self.failureException('Async operation timed out after %s seconds' % timeout)
                    except Exception:
                        self.__failure = sys.exc_info()
                    self.stop()
                self.__timeout = self.io_loop.add_timeout(self.io_loop.time() + timeout, timeout_func)
            while True:
                self.__running = True
                self.io_loop.start()
                if self.__failure is not None or condition is None or condition():
                    break
            if self.__timeout is not None:
                self.io_loop.remove_timeout(self.__timeout)
                self.__timeout = None
        assert self.__stopped
        self.__stopped = False
        self.__rethrow()
        result = self.__stop_args
        self.__stop_args = None
        return result