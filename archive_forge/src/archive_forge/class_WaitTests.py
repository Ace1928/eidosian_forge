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
class WaitTests(TestCase):
    """
    Tests for wait_for decorators.
    """

    def setUp(self):
        self.reactor = FakeReactor()
        self.eventloop = EventLoop(lambda: self.reactor, lambda f, g: None)
        self.eventloop.no_setup()
    DECORATOR_CALL = 'wait_for(timeout=5)'

    def decorator(self):
        return lambda func: self.eventloop.wait_for(timeout=5)(func)

    def make_wrapped_function(self):
        """
        Return a function wrapped with the decorator being tested that returns
        its first argument, or raises it if it's an exception.
        """
        decorator = self.decorator()

        @decorator
        def passthrough(argument):
            if isinstance(argument, Exception):
                raise argument
            return argument
        return passthrough

    def test_name(self):
        """
        The function decorated with the wait decorator has the same name as the
        original function.
        """
        decorator = self.decorator()

        @decorator
        def some_name(argument):
            pass
        self.assertEqual(some_name.__name__, 'some_name')

    def test_signature(self):
        """
        The function decorated with the wait decorator has the same signature
        as the original function.
        """
        decorator = self.decorator()

        def some_name(arg1, arg2, karg1=2, *args, **kw):
            pass
        decorated = decorator(some_name)
        self.assertEqual(inspect.signature(some_name), inspect.signature(decorated))

    def test_wrapped_function(self):
        """
        The function wrapped by the wait decorator can be accessed via the
        `__wrapped__` attribute.
        """
        decorator = self.decorator()

        def func():
            pass
        wrapper = decorator(func)
        self.assertIdentical(wrapper.__wrapped__, func)

    def test_reactor_thread_disallowed(self):
        """
        Functions decorated with the wait decorator cannot be called from the
        reactor thread.
        """
        self.patch(threadable, 'isInIOThread', lambda: True)
        f = self.make_wrapped_function()
        self.assertRaises(RuntimeError, f, None)

    def test_wait_for_reactor_thread(self):
        """
        The function decorated with the wait decorator is run in the reactor
        thread.
        """
        in_call_from_thread = []
        decorator = self.decorator()

        @decorator
        def func():
            in_call_from_thread.append(self.reactor.in_call_from_thread)
        in_call_from_thread.append(self.reactor.in_call_from_thread)
        func()
        in_call_from_thread.append(self.reactor.in_call_from_thread)
        self.assertEqual(in_call_from_thread, [False, True, False])

    def test_arguments(self):
        """
        The function decorated with wait decorator gets all arguments passed
        to the wrapper.
        """
        calls = []
        decorator = self.decorator()

        @decorator
        def func(a, b, c):
            calls.append((a, b, c))
        func(1, 2, c=3)
        self.assertEqual(calls, [(1, 2, 3)])

    def test_classmethod(self):
        """
        The function decorated with the wait decorator can be a classmethod.
        """
        calls = []
        decorator = self.decorator()

        class C(object):

            @decorator
            @classmethod
            def func(cls, a, b, c):
                calls.append((a, b, c))

            @classmethod
            @decorator
            def func2(cls, a, b, c):
                calls.append((a, b, c))
        C.func(1, 2, c=3)
        C.func2(1, 2, c=3)
        self.assertEqual(calls, [(1, 2, 3), (1, 2, 3)])

    def test_deferred_success_result(self):
        """
        If the underlying function returns a Deferred, the wrapper returns a
        the Deferred's result.
        """
        passthrough = self.make_wrapped_function()
        result = passthrough(succeed(123))
        self.assertEqual(result, 123)

    def test_deferred_failure_result(self):
        """
        If the underlying function returns a Deferred with an errback, the
        wrapper throws an exception.
        """
        passthrough = self.make_wrapped_function()
        self.assertRaises(ZeroDivisionError, passthrough, fail(ZeroDivisionError()))

    def test_regular_result(self):
        """
        If the underlying function returns a non-Deferred, the wrapper returns
        that result.
        """
        passthrough = self.make_wrapped_function()
        result = passthrough(123)
        self.assertEqual(result, 123)

    def test_exception_result(self):
        """
        If the underlying function throws an exception, the wrapper raises
        that exception.
        """
        raiser = self.make_wrapped_function()
        self.assertRaises(ZeroDivisionError, raiser, ZeroDivisionError())

    def test_control_c_is_possible(self):
        """
        A call to a decorated function responds to a Ctrl-C (i.e. with a
        KeyboardInterrupt) in a timely manner.
        """
        if platform.type != 'posix':
            raise SkipTest("I don't have the energy to fight Windows semantics.")
        program = "import os, threading, signal, time, sys\nimport crochet\ncrochet.setup()\nfrom twisted.internet.defer import Deferred\n\nif sys.platform.startswith('win'):\n    signal.signal(signal.SIGBREAK, signal.default_int_handler)\n    sig_int=signal.CTRL_BREAK_EVENT\n    sig_kill=signal.SIGTERM\nelse:\n    sig_int=signal.SIGINT\n    sig_kill=signal.SIGKILL\n\n\ndef interrupt():\n    time.sleep(0.1) # Make sure we've hit wait()\n    os.kill(os.getpid(), sig_int)\n    time.sleep(1)\n    # Still running, test shall fail...\n    os.kill(os.getpid(), sig_kill)\n\nt = threading.Thread(target=interrupt, daemon=True)\nt.start()\n\n@crochet.%s\ndef wait():\n    return Deferred()\n\ntry:\n    wait()\nexcept KeyboardInterrupt:\n    sys.exit(23)\n" % (self.DECORATOR_CALL,)
        kw = {'cwd': crochet_directory}
        if platform.type.startswith('win'):
            kw['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        process = subprocess.Popen([sys.executable, '-c', program], **kw)
        self.assertEqual(process.wait(), 23)

    def test_reactor_stop_unblocks(self):
        """
        Any @wait_for_reactor-decorated calls still waiting when the reactor
        has stopped will get a ReactorStopped exception.
        """
        program = 'import os, threading, signal, time, sys\n\nfrom twisted.internet.defer import Deferred\nfrom twisted.internet import reactor\n\nimport crochet\ncrochet.setup()\n\n@crochet.%s\ndef run():\n    reactor.callLater(0.1, reactor.stop)\n    return Deferred()\n\ntry:\n    er = run()\nexcept crochet.ReactorStopped:\n    sys.exit(23)\n' % (self.DECORATOR_CALL,)
        process = subprocess.Popen([sys.executable, '-c', program], cwd=crochet_directory)
        self.assertEqual(process.wait(), 23)

    def test_timeoutRaises(self):
        """
        If a function wrapped with wait_for hits the timeout, it raises
        TimeoutError.
        """

        @self.eventloop.wait_for(timeout=0.5)
        def times_out():
            return Deferred().addErrback(lambda f: f.trap(CancelledError))
        start = time.time()
        self.assertRaises(TimeoutError, times_out)
        self.assertTrue(abs(time.time() - start - 0.5) < 0.1)

    def test_timeoutCancels(self):
        """
        If a function wrapped with wait_for hits the timeout, it cancels
        the underlying Deferred.
        """
        result = Deferred()
        error = []
        result.addErrback(error.append)

        @self.eventloop.wait_for(timeout=0.0)
        def times_out():
            return result
        self.assertRaises(TimeoutError, times_out)
        self.assertIsInstance(error[0].value, CancelledError)

    def test_async_function(self):
        """
        Async functions can be wrapped with @wait_for.
        """

        @self.eventloop.wait_for(timeout=0.1)
        async def go():
            self.assertTrue(self.reactor.in_call_from_thread)
            return 17
        self.assertEqual((go(), go()), (17, 17))
        self.assertFalse(inspect.iscoroutinefunction(go))