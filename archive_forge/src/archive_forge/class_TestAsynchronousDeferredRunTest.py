import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
class TestAsynchronousDeferredRunTest(NeedsTwistedTestCase):

    def make_reactor(self):
        from twisted.internet import reactor
        return reactor

    def make_result(self):
        return ExtendedTestResult()

    def make_runner(self, test, timeout=None, suppress_twisted_logging=True, store_twisted_logs=True):
        if timeout is None:
            timeout = self.make_timeout()
        return AsynchronousDeferredRunTest(test, test.exception_handlers, timeout=timeout, suppress_twisted_logging=suppress_twisted_logging, store_twisted_logs=store_twisted_logs)

    def make_timeout(self):
        return 0.005

    def test_setUp_returns_deferred_that_fires_later(self):
        call_log = []
        marker = object()
        d = defer.Deferred().addCallback(call_log.append)

        class SomeCase(TestCase):

            def setUp(self):
                super().setUp()
                call_log.append('setUp')
                return d

            def test_something(self):
                call_log.append('test')

        def fire_deferred():
            self.assertThat(call_log, Equals(['setUp']))
            d.callback(marker)
        test = SomeCase('test_something')
        timeout = self.make_timeout()
        runner = self.make_runner(test, timeout=timeout)
        result = self.make_result()
        reactor = self.make_reactor()
        reactor.callLater(timeout, fire_deferred)
        runner.run(result)
        self.assertThat(call_log, Equals(['setUp', marker, 'test']))

    def test_calls_setUp_test_tearDown_in_sequence(self):
        call_log = []
        a = defer.Deferred()
        a.addCallback(lambda x: call_log.append('a'))
        b = defer.Deferred()
        b.addCallback(lambda x: call_log.append('b'))
        c = defer.Deferred()
        c.addCallback(lambda x: call_log.append('c'))

        class SomeCase(TestCase):

            def setUp(self):
                super().setUp()
                call_log.append('setUp')
                return a

            def test_success(self):
                call_log.append('test')
                return b

            def tearDown(self):
                super().tearDown()
                call_log.append('tearDown')
                return c
        test = SomeCase('test_success')
        timeout = self.make_timeout()
        runner = self.make_runner(test, timeout)
        result = self.make_result()
        reactor = self.make_reactor()

        def fire_a():
            self.assertThat(call_log, Equals(['setUp']))
            a.callback(None)

        def fire_b():
            self.assertThat(call_log, Equals(['setUp', 'a', 'test']))
            b.callback(None)

        def fire_c():
            self.assertThat(call_log, Equals(['setUp', 'a', 'test', 'b', 'tearDown']))
            c.callback(None)
        reactor.callLater(timeout * 0.25, fire_a)
        reactor.callLater(timeout * 0.5, fire_b)
        reactor.callLater(timeout * 0.75, fire_c)
        runner.run(result)
        self.assertThat(call_log, Equals(['setUp', 'a', 'test', 'b', 'tearDown', 'c']))

    def test_async_cleanups(self):

        class SomeCase(TestCase):

            def test_whatever(self):
                pass
        test = SomeCase('test_whatever')
        call_log = []
        a = defer.Deferred().addCallback(lambda x: call_log.append('a'))
        b = defer.Deferred().addCallback(lambda x: call_log.append('b'))
        c = defer.Deferred().addCallback(lambda x: call_log.append('c'))
        test.addCleanup(lambda: a)
        test.addCleanup(lambda: b)
        test.addCleanup(lambda: c)

        def fire_a():
            self.assertThat(call_log, Equals([]))
            a.callback(None)

        def fire_b():
            self.assertThat(call_log, Equals(['a']))
            b.callback(None)

        def fire_c():
            self.assertThat(call_log, Equals(['a', 'b']))
            c.callback(None)
        timeout = self.make_timeout()
        reactor = self.make_reactor()
        reactor.callLater(timeout * 0.25, fire_a)
        reactor.callLater(timeout * 0.5, fire_b)
        reactor.callLater(timeout * 0.75, fire_c)
        runner = self.make_runner(test, timeout)
        result = self.make_result()
        runner.run(result)
        self.assertThat(call_log, Equals(['a', 'b', 'c']))

    def test_clean_reactor(self):
        reactor = self.make_reactor()
        timeout = self.make_timeout()

        class SomeCase(TestCase):

            def test_cruft(self):
                reactor.callLater(timeout * 10.0, lambda: None)
        test = SomeCase('test_cruft')
        runner = self.make_runner(test, timeout)
        result = self.make_result()
        runner.run(result)
        self.assertThat([event[:2] for event in result._events], Equals([('startTest', test), ('addError', test), ('stopTest', test)]))
        error = result._events[1][2]
        self.assertThat(error, KeysEqual('traceback', 'twisted-log'))

    def test_exports_reactor(self):
        reactor = self.make_reactor()
        timeout = self.make_timeout()

        class SomeCase(TestCase):

            def test_cruft(self):
                self.assertIs(reactor, self.reactor)
        test = SomeCase('test_cruft')
        runner = self.make_runner(test, timeout)
        result = TestResult()
        runner.run(result)
        self.assertEqual([], result.errors)
        self.assertEqual([], result.failures)

    def test_unhandled_error_from_deferred(self):
        self.useFixture(DebugTwisted(False))

        class SomeCase(TestCase):

            def test_cruft(self):
                defer.maybeDeferred(lambda: 1 / 0)
                defer.maybeDeferred(lambda: 2 / 0)
        test = SomeCase('test_cruft')
        runner = self.make_runner(test)
        result = self.make_result()
        runner.run(result)
        error = result._events[1][2]
        result._events[1] = ('addError', test, None)
        self.assertThat(result._events, Equals([('startTest', test), ('addError', test, None), ('stopTest', test)]))
        self.assertThat(error, KeysEqual('twisted-log', 'unhandled-error-in-deferred', 'unhandled-error-in-deferred-1'))

    def test_unhandled_error_from_deferred_combined_with_error(self):
        self.useFixture(DebugTwisted(False))

        class SomeCase(TestCase):

            def test_cruft(self):
                defer.maybeDeferred(lambda: 1 / 0)
                2 / 0
        test = SomeCase('test_cruft')
        runner = self.make_runner(test)
        result = self.make_result()
        runner.run(result)
        error = result._events[1][2]
        result._events[1] = ('addError', test, None)
        self.assertThat(result._events, Equals([('startTest', test), ('addError', test, None), ('stopTest', test)]))
        self.assertThat(error, KeysEqual('traceback', 'twisted-log', 'unhandled-error-in-deferred'))

    @skipIf(os.name != 'posix', 'Sending SIGINT with os.kill is posix only')
    def test_keyboard_interrupt_stops_test_run(self):
        SIGINT = getattr(signal, 'SIGINT', None)
        if not SIGINT:
            raise self.skipTest('SIGINT unavailable')

        class SomeCase(TestCase):

            def test_pause(self):
                return defer.Deferred()
        test = SomeCase('test_pause')
        reactor = self.make_reactor()
        timeout = self.make_timeout()
        runner = self.make_runner(test, timeout * 5)
        result = self.make_result()
        reactor.callLater(timeout, os.kill, os.getpid(), SIGINT)
        runner.run(result)
        self.assertThat(result.shouldStop, Equals(True))

    @skipIf(os.name != 'posix', 'Sending SIGINT with os.kill is posix only')
    def test_fast_keyboard_interrupt_stops_test_run(self):
        SIGINT = getattr(signal, 'SIGINT', None)
        if not SIGINT:
            raise self.skipTest('SIGINT unavailable')

        class SomeCase(TestCase):

            def test_pause(self):
                return defer.Deferred()
        test = SomeCase('test_pause')
        reactor = self.make_reactor()
        timeout = self.make_timeout()
        runner = self.make_runner(test, timeout * 5)
        result = self.make_result()
        reactor.callWhenRunning(os.kill, os.getpid(), SIGINT)
        runner.run(result)
        self.assertThat(result.shouldStop, Equals(True))

    def test_timeout_causes_test_error(self):

        class SomeCase(TestCase):

            def test_pause(self):
                return defer.Deferred()
        test = SomeCase('test_pause')
        runner = self.make_runner(test)
        result = self.make_result()
        runner.run(result)
        error = result._events[1][2]
        self.assertThat([event[:2] for event in result._events], Equals([('startTest', test), ('addError', test), ('stopTest', test)]))
        self.assertIn('TimeoutError', str(error['traceback']))

    def test_convenient_construction(self):
        reactor = object()
        timeout = object()
        handler = object()
        factory = AsynchronousDeferredRunTest.make_factory(reactor, timeout)
        runner = factory(self, [handler])
        self.assertIs(reactor, runner._reactor)
        self.assertIs(timeout, runner._timeout)
        self.assertIs(self, runner.case)
        self.assertEqual([handler], runner.handlers)

    def test_use_convenient_factory(self):
        factory = AsynchronousDeferredRunTest.make_factory()

        class SomeCase(TestCase):
            run_tests_with = factory

            def test_something(self):
                pass
        case = SomeCase('test_something')
        case.run()

    def test_convenient_construction_default_reactor(self):
        reactor = object()
        handler = object()
        factory = AsynchronousDeferredRunTest.make_factory(reactor=reactor)
        runner = factory(self, [handler])
        self.assertIs(reactor, runner._reactor)
        self.assertIs(self, runner.case)
        self.assertEqual([handler], runner.handlers)

    def test_convenient_construction_default_timeout(self):
        timeout = object()
        handler = object()
        factory = AsynchronousDeferredRunTest.make_factory(timeout=timeout)
        runner = factory(self, [handler])
        self.assertIs(timeout, runner._timeout)
        self.assertIs(self, runner.case)
        self.assertEqual([handler], runner.handlers)

    def test_convenient_construction_default_debugging(self):
        handler = object()
        factory = AsynchronousDeferredRunTest.make_factory(debug=True)
        runner = factory(self, [handler])
        self.assertIs(self, runner.case)
        self.assertEqual([handler], runner.handlers)
        self.assertEqual(True, runner._debug)

    def test_deferred_error(self):

        class SomeTest(TestCase):

            def test_something(self):
                return defer.maybeDeferred(lambda: 1 / 0)
        test = SomeTest('test_something')
        runner = self.make_runner(test)
        result = self.make_result()
        runner.run(result)
        self.assertThat([event[:2] for event in result._events], Equals([('startTest', test), ('addError', test), ('stopTest', test)]))
        error = result._events[1][2]
        self.assertThat(error, KeysEqual('traceback', 'twisted-log'))

    def test_only_addError_once(self):
        self.useFixture(DebugTwisted(False))
        reactor = self.make_reactor()

        class WhenItRains(TestCase):

            def it_pours(self):
                self.addCleanup(lambda: 3 / 0)
                from twisted.internet.protocol import ServerFactory
                reactor.listenTCP(0, ServerFactory(), interface='127.0.0.1')
                defer.maybeDeferred(lambda: 2 / 0)
                raise RuntimeError('Excess precipitation')
        test = WhenItRains('it_pours')
        runner = self.make_runner(test)
        result = self.make_result()
        runner.run(result)
        self.assertThat([event[:2] for event in result._events], Equals([('startTest', test), ('addError', test), ('stopTest', test)]))
        error = result._events[1][2]
        self.assertThat(error, KeysEqual('traceback', 'traceback-1', 'traceback-2', 'twisted-log', 'unhandled-error-in-deferred'))

    def test_log_err_is_error(self):

        class LogAnError(TestCase):

            def test_something(self):
                try:
                    1 / 0
                except ZeroDivisionError:
                    f = failure.Failure()
                log.err(f)
        test = LogAnError('test_something')
        runner = self.make_runner(test, store_twisted_logs=False)
        result = self.make_result()
        runner.run(result)
        self.assertThat(result._events, MatchesEvents(('startTest', test), ('addError', test, {'logged-error': AsText(ContainsAll(['Traceback (most recent call last):', 'ZeroDivisionError']))}), ('stopTest', test)))

    def test_log_err_flushed_is_success(self):

        class LogAnError(TestCase):

            def test_something(self):
                try:
                    1 / 0
                except ZeroDivisionError:
                    f = failure.Failure()
                log.err(f)
                flush_logged_errors(ZeroDivisionError)
        test = LogAnError('test_something')
        runner = self.make_runner(test, store_twisted_logs=False)
        result = self.make_result()
        runner.run(result)
        self.assertThat(result._events, MatchesEvents(('startTest', test), ('addSuccess', test), ('stopTest', test)))

    def test_log_in_details(self):

        class LogAnError(TestCase):

            def test_something(self):
                log.msg('foo')
                1 / 0
        test = LogAnError('test_something')
        runner = self.make_runner(test, store_twisted_logs=True)
        result = self.make_result()
        runner.run(result)
        self.assertThat(result._events, MatchesEvents(('startTest', test), ('addError', test, {'traceback': Not(Is(None)), 'twisted-log': AsText(EndsWith(' foo\n'))}), ('stopTest', test)))

    def test_do_not_log_to_twisted(self):
        messages = []
        publisher, _ = _get_global_publisher_and_observers()
        publisher.addObserver(messages.append)
        self.addCleanup(publisher.removeObserver, messages.append)

        class LogSomething(TestCase):

            def test_something(self):
                log.msg('foo')
        test = LogSomething('test_something')
        runner = self.make_runner(test, suppress_twisted_logging=True)
        result = self.make_result()
        runner.run(result)
        self.assertThat(messages, Equals([]))

    def test_log_to_twisted(self):
        messages = []
        publisher, _ = _get_global_publisher_and_observers()
        publisher.addObserver(messages.append)

        class LogSomething(TestCase):

            def test_something(self):
                log.msg('foo')
        test = LogSomething('test_something')
        runner = self.make_runner(test, suppress_twisted_logging=False)
        result = self.make_result()
        runner.run(result)
        self.assertThat(messages, MatchesListwise([ContainsDict({'message': Equals(('foo',))})]))

    def test_restore_observers(self):
        publisher, observers = _get_global_publisher_and_observers()

        class LogSomething(TestCase):

            def test_something(self):
                pass
        test = LogSomething('test_something')
        runner = self.make_runner(test)
        result = self.make_result()
        runner.run(result)
        self.assertThat(_get_global_publisher_and_observers()[1], Equals(observers))

    def test_restore_observers_after_timeout(self):
        publisher, observers = _get_global_publisher_and_observers()

        class LogSomething(TestCase):

            def test_something(self):
                return defer.Deferred()
        test = LogSomething('test_something')
        runner = self.make_runner(test, timeout=0.0001)
        result = self.make_result()
        runner.run(result)
        self.assertThat(_get_global_publisher_and_observers()[1], Equals(observers))

    def test_debugging_unchanged_during_test_by_default(self):
        debugging = [(defer.Deferred.debug, DelayedCall.debug)]

        class SomeCase(TestCase):

            def test_debugging_enabled(self):
                debugging.append((defer.Deferred.debug, DelayedCall.debug))
        test = SomeCase('test_debugging_enabled')
        runner = AsynchronousDeferredRunTest(test, handlers=test.exception_handlers, reactor=self.make_reactor(), timeout=self.make_timeout())
        runner.run(self.make_result())
        self.assertEqual(debugging[0], debugging[1])

    def test_debugging_enabled_during_test_with_debug_flag(self):
        self.patch(defer.Deferred, 'debug', False)
        self.patch(DelayedCall, 'debug', False)
        debugging = []

        class SomeCase(TestCase):

            def test_debugging_enabled(self):
                debugging.append((defer.Deferred.debug, DelayedCall.debug))
        test = SomeCase('test_debugging_enabled')
        runner = AsynchronousDeferredRunTest(test, handlers=test.exception_handlers, reactor=self.make_reactor(), timeout=self.make_timeout(), debug=True)
        runner.run(self.make_result())
        self.assertEqual([(True, True)], debugging)
        self.assertEqual(False, defer.Deferred.debug)
        self.assertEqual(False, defer.Deferred.debug)