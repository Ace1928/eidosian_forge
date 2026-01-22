import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
class TestRunInReactor(NeedsTwistedTestCase):

    def make_reactor(self):
        from twisted.internet import reactor
        return reactor

    def make_spinner(self, reactor=None):
        if reactor is None:
            reactor = self.make_reactor()
        return _spinner.Spinner(reactor)

    def make_timeout(self):
        return 0.01

    def test_function_called(self):
        calls = []
        marker = object()
        self.make_spinner().run(self.make_timeout(), calls.append, marker)
        self.assertThat(calls, Equals([marker]))

    def test_return_value_returned(self):
        marker = object()
        result = self.make_spinner().run(self.make_timeout(), lambda: marker)
        self.assertThat(result, Is(marker))

    def test_exception_reraised(self):
        self.assertThat(lambda: self.make_spinner().run(self.make_timeout(), lambda: 1 / 0), Raises(MatchesException(ZeroDivisionError)))

    def test_keyword_arguments(self):
        calls = []

        def function(*a, **kw):
            return calls.extend([a, kw])
        self.make_spinner().run(self.make_timeout(), function, foo=42)
        self.assertThat(calls, Equals([(), {'foo': 42}]))

    def test_not_reentrant(self):
        spinner = self.make_spinner()
        self.assertThat(lambda: spinner.run(self.make_timeout(), spinner.run, self.make_timeout(), lambda: None), Raises(MatchesException(_spinner.ReentryError)))

    def test_deferred_value_returned(self):
        marker = object()
        result = self.make_spinner().run(self.make_timeout(), lambda: defer.succeed(marker))
        self.assertThat(result, Is(marker))

    def test_preserve_signal_handler(self):
        signals = ['SIGINT', 'SIGTERM', 'SIGCHLD']
        signals = list(filter(None, (getattr(signal, name, None) for name in signals)))
        for sig in signals:
            self.addCleanup(signal.signal, sig, signal.getsignal(sig))
        new_hdlrs = list((lambda *a: None for _ in signals))
        for sig, hdlr in zip(signals, new_hdlrs):
            signal.signal(sig, hdlr)
        spinner = self.make_spinner()
        spinner.run(self.make_timeout(), lambda: None)
        self.assertItemsEqual(new_hdlrs, list(map(signal.getsignal, signals)))

    def test_timeout(self):
        timeout = self.make_timeout()
        self.assertThat(lambda: self.make_spinner().run(timeout, lambda: defer.Deferred()), Raises(MatchesException(_spinner.TimeoutError)))

    def test_no_junk_by_default(self):
        spinner = self.make_spinner()
        self.assertThat(spinner.get_junk(), Equals([]))

    def test_clean_do_nothing(self):
        spinner = self.make_spinner()
        result = spinner._clean()
        self.assertThat(result, Equals([]))

    def test_clean_delayed_call(self):
        reactor = self.make_reactor()
        spinner = self.make_spinner(reactor)
        call = reactor.callLater(10, lambda: None)
        results = spinner._clean()
        self.assertThat(results, Equals([call]))
        self.assertThat(call.active(), Equals(False))

    def test_clean_delayed_call_cancelled(self):
        reactor = self.make_reactor()
        spinner = self.make_spinner(reactor)
        call = reactor.callLater(10, lambda: None)
        call.cancel()
        results = spinner._clean()
        self.assertThat(results, Equals([]))

    def test_clean_selectables(self):
        from twisted.internet.protocol import ServerFactory
        reactor = self.make_reactor()
        spinner = self.make_spinner(reactor)
        port = reactor.listenTCP(0, ServerFactory(), interface='127.0.0.1')
        spinner.run(self.make_timeout(), lambda: None)
        results = spinner.get_junk()
        self.assertThat(results, Equals([port]))

    def test_clean_running_threads(self):
        import threading
        import time
        current_threads = list(threading.enumerate())
        reactor = self.make_reactor()
        timeout = self.make_timeout()
        spinner = self.make_spinner(reactor)
        spinner.run(timeout, reactor.callInThread, time.sleep, timeout / 2.0)
        self.assertThat(list(threading.enumerate()), Equals(current_threads))

    def test_leftover_junk_available(self):
        from twisted.internet.protocol import ServerFactory
        reactor = self.make_reactor()
        spinner = self.make_spinner(reactor)
        port = spinner.run(self.make_timeout(), reactor.listenTCP, 0, ServerFactory(), interface='127.0.0.1')
        self.assertThat(spinner.get_junk(), Equals([port]))

    def test_will_not_run_with_previous_junk(self):
        from twisted.internet.protocol import ServerFactory
        reactor = self.make_reactor()
        spinner = self.make_spinner(reactor)
        timeout = self.make_timeout()
        spinner.run(timeout, reactor.listenTCP, 0, ServerFactory(), interface='127.0.0.1')
        self.assertThat(lambda: spinner.run(timeout, lambda: None), Raises(MatchesException(_spinner.StaleJunkError)))

    def test_clear_junk_clears_previous_junk(self):
        from twisted.internet.protocol import ServerFactory
        reactor = self.make_reactor()
        spinner = self.make_spinner(reactor)
        timeout = self.make_timeout()
        port = spinner.run(timeout, reactor.listenTCP, 0, ServerFactory(), interface='127.0.0.1')
        junk = spinner.clear_junk()
        self.assertThat(junk, Equals([port]))
        self.assertThat(spinner.get_junk(), Equals([]))

    @skipIf(os.name != 'posix', 'Sending SIGINT with os.kill is posix only')
    def test_sigint_raises_no_result_error(self):
        SIGINT = getattr(signal, 'SIGINT', None)
        if not SIGINT:
            self.skipTest('SIGINT not available')
        reactor = self.make_reactor()
        spinner = self.make_spinner(reactor)
        timeout = self.make_timeout()
        reactor.callLater(timeout, os.kill, os.getpid(), SIGINT)
        self.assertThat(lambda: spinner.run(timeout * 5, defer.Deferred), Raises(MatchesException(_spinner.NoResultError)))
        self.assertEqual([], spinner._clean())

    @skipIf(os.name != 'posix', 'Sending SIGINT with os.kill is posix only')
    def test_sigint_raises_no_result_error_second_time(self):
        self.test_sigint_raises_no_result_error()

    @skipIf(os.name != 'posix', 'Sending SIGINT with os.kill is posix only')
    def test_fast_sigint_raises_no_result_error(self):
        SIGINT = getattr(signal, 'SIGINT', None)
        if not SIGINT:
            self.skipTest('SIGINT not available')
        reactor = self.make_reactor()
        spinner = self.make_spinner(reactor)
        timeout = self.make_timeout()
        reactor.callWhenRunning(os.kill, os.getpid(), SIGINT)
        self.assertThat(lambda: spinner.run(timeout * 5, defer.Deferred), Raises(MatchesException(_spinner.NoResultError)))
        self.assertEqual([], spinner._clean())

    @skipIf(os.name != 'posix', 'Sending SIGINT with os.kill is posix only')
    def test_fast_sigint_raises_no_result_error_second_time(self):
        self.test_fast_sigint_raises_no_result_error()

    def test_fires_after_timeout(self):
        reactor = self.make_reactor()
        spinner1 = self.make_spinner(reactor)
        timeout = self.make_timeout()
        deferred1 = defer.Deferred()
        self.expectThat(lambda: spinner1.run(timeout, lambda: deferred1), Raises(MatchesException(_spinner.TimeoutError)))
        marker = object()
        deferred2 = defer.Deferred()
        deferred1.addCallback(lambda ignored: reactor.callLater(0, deferred2.callback, marker))

        def fire_other():
            """Fire Deferred from the last spin while waiting for this one."""
            deferred1.callback(object())
            return deferred2
        spinner2 = self.make_spinner(reactor)
        self.assertThat(spinner2.run(timeout, fire_other), Is(marker))