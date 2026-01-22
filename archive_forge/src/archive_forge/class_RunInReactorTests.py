from __future__ import absolute_import
import threading
import subprocess
import time
import gc
import sys
import weakref
import tempfile
import os
import inspect
from unittest import SkipTest
from twisted.trial.unittest import TestCase
from twisted.internet.defer import succeed, Deferred, fail, CancelledError
from twisted.python.failure import Failure
from twisted.python import threadable
from twisted.python.runtime import platform
from .._eventloop import (
from .test_setup import FakeReactor
from .. import (
from ..tests import crochet_directory
import os, threading, signal, time, sys
import crochet
from twisted.internet.defer import Deferred
import os, threading, signal, time, sys
from twisted.internet.defer import Deferred
from twisted.internet import reactor
import crochet
import os, threading, signal, time, sys
from twisted.internet.defer import Deferred
from twisted.internet import reactor
import crochet
import os, threading, signal, time, sys
from twisted.internet.defer import Deferred, CancelledError
import crochet
from crochet import EventualResult
from twisted.internet.defer import Deferred
import os, threading, signal, time, sys
import crochet
from twisted.internet.defer import Deferred
import os, threading, signal, time, sys
from twisted.internet.defer import Deferred
from twisted.internet import reactor
import crochet
class RunInReactorTests(TestCase):
    """
    Tests for the run_in_reactor decorator.
    """

    def test_signature(self):
        """
        The function decorated with the run_in_reactor decorator has the same
        signature as the original function.
        """
        c = EventLoop(lambda: FakeReactor(), lambda f, g: None)

        def some_name(arg1, arg2, karg1=2, *args, **kw):
            pass
        decorated = c.run_in_reactor(some_name)
        self.assertEqual(inspect.signature(some_name), inspect.signature(decorated))

    def test_name(self):
        """
        The function decorated with run_in_reactor has the same name as the
        original function.
        """
        c = EventLoop(lambda: FakeReactor(), lambda f, g: None)

        @c.run_in_reactor
        def some_name():
            pass
        self.assertEqual(some_name.__name__, 'some_name')

    def test_run_in_reactor_thread(self):
        """
        The function decorated with run_in_reactor is run in the reactor
        thread.
        """
        myreactor = FakeReactor()
        c = EventLoop(lambda: myreactor, lambda f, g: None)
        c.no_setup()
        calls = []

        @c.run_in_reactor
        def func(a, b, c):
            self.assertTrue(myreactor.in_call_from_thread)
            calls.append((a, b, c))
        func(1, 2, c=3)
        self.assertEqual(calls, [(1, 2, 3)])

    def test_method(self):
        """
        The function decorated with the wait decorator can be a method.
        """
        myreactor = FakeReactor()
        c = EventLoop(lambda: myreactor, lambda f, g: None)
        c.no_setup()
        calls = []

        class C(object):

            @c.run_in_reactor
            def func(self, a, b, c):
                calls.append((self, a, b, c))
        o = C()
        o.func(1, 2, c=3)
        self.assertEqual(calls, [(o, 1, 2, 3)])

    def test_classmethod(self):
        """
        The function decorated with the wait decorator can be a classmethod.
        """
        myreactor = FakeReactor()
        c = EventLoop(lambda: myreactor, lambda f, g: None)
        c.no_setup()
        calls = []

        class C(object):

            @c.run_in_reactor
            @classmethod
            def func(cls, a, b, c):
                calls.append((cls, a, b, c))

            @classmethod
            @c.run_in_reactor
            def func2(cls, a, b, c):
                calls.append((cls, a, b, c))
        C.func(1, 2, c=3)
        C.func2(1, 2, c=3)
        self.assertEqual(calls, [(C, 1, 2, 3), (C, 1, 2, 3)])

    def test_wrap_method(self):
        """
        The object decorated with the wait decorator can be a method object
        """
        myreactor = FakeReactor()
        c = EventLoop(lambda: myreactor, lambda f, g: None)
        c.no_setup()
        calls = []

        class C(object):

            def func(self, a, b, c):
                calls.append((a, b, c))
        f = c.run_in_reactor(C().func)
        f(4, 5, c=6)
        self.assertEqual(calls, [(4, 5, 6)])

    def make_wrapped_function(self):
        """
        Return a function wrapped with run_in_reactor that returns its first
        argument.
        """
        myreactor = FakeReactor()
        c = EventLoop(lambda: myreactor, lambda f, g: None)
        c.no_setup()

        @c.run_in_reactor
        def passthrough(argument):
            return argument
        return passthrough

    def test_deferred_success_result(self):
        """
        If the underlying function returns a Deferred, the wrapper returns a
        EventualResult hooked up to the Deferred.
        """
        passthrough = self.make_wrapped_function()
        result = passthrough(succeed(123))
        self.assertIsInstance(result, EventualResult)
        self.assertEqual(result.wait(0.1), 123)

    def test_deferred_failure_result(self):
        """
        If the underlying function returns a Deferred, the wrapper returns a
        EventualResult hooked up to the Deferred that can deal with failures
        as well.
        """
        passthrough = self.make_wrapped_function()
        result = passthrough(fail(ZeroDivisionError()))
        self.assertIsInstance(result, EventualResult)
        self.assertRaises(ZeroDivisionError, result.wait, 0.1)

    def test_regular_result(self):
        """
        If the underlying function returns a non-Deferred, the wrapper returns
        a EventualResult hooked up to a Deferred wrapping the result.
        """
        passthrough = self.make_wrapped_function()
        result = passthrough(123)
        self.assertIsInstance(result, EventualResult)
        self.assertEqual(result.wait(0.1), 123)

    def test_exception_result(self):
        """
        If the underlying function throws an exception, the wrapper returns a
        EventualResult hooked up to a Deferred wrapping the exception.
        """
        myreactor = FakeReactor()
        c = EventLoop(lambda: myreactor, lambda f, g: None)
        c.no_setup()

        @c.run_in_reactor
        def raiser():
            1 / 0
        result = raiser()
        self.assertIsInstance(result, EventualResult)
        self.assertRaises(ZeroDivisionError, result.wait, 0.1)

    def test_registry(self):
        """
        @run_in_reactor registers the EventualResult in the ResultRegistry.
        """
        myreactor = FakeReactor()
        c = EventLoop(lambda: myreactor, lambda f, g: None)
        c.no_setup()

        @c.run_in_reactor
        def run():
            return
        result = run()
        self.assertIn(result, c._registry._results)

    def test_wrapped_function(self):
        """
        The function wrapped by @run_in_reactor can be accessed via the
        `__wrapped__` attribute.
        """
        c = EventLoop(lambda: None, lambda f, g: None)

        def func():
            pass
        wrapper = c.run_in_reactor(func)
        self.assertIdentical(wrapper.__wrapped__, func)

    def test_async_function(self):
        """
        Async functions can be wrapped with @run_in_reactor.
        """
        myreactor = FakeReactor()
        c = EventLoop(lambda: myreactor, lambda f, g: None)
        c.no_setup()
        calls = []

        @c.run_in_reactor
        async def go():
            self.assertTrue(myreactor.in_call_from_thread)
            calls.append(1)
            return 23
        self.assertEqual((go().wait(0.1), go().wait(0.1)), (23, 23))
        self.assertEqual(len(calls), 2)
        self.assertFalse(inspect.iscoroutinefunction(go))