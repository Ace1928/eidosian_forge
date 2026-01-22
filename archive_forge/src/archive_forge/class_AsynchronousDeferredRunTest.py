import io
import warnings
import sys
from fixtures import CompoundFixture, Fixture
from testtools.content import Content, text_content
from testtools.content_type import UTF8_TEXT
from testtools.runtest import RunTest, _raise_force_fail_error
from ._deferred import extract_result
from ._spinner import (
from twisted.internet import defer
from twisted.python import log
class AsynchronousDeferredRunTest(_DeferredRunTest):
    """Runner for tests that return Deferreds that fire asynchronously.

    Use this runner when you have tests that return Deferreds that will
    only fire if the reactor is left to spin for a while.
    """

    def __init__(self, case, handlers=None, last_resort=None, reactor=None, timeout=0.005, debug=False, suppress_twisted_logging=True, store_twisted_logs=True):
        """Construct an ``AsynchronousDeferredRunTest``.

        Please be sure to always use keyword syntax, not positional, as the
        base class may add arguments in future - and for core code
        compatibility with that we have to insert them before the local
        parameters.

        :param TestCase case: The `TestCase` to run.
        :param handlers: A list of exception handlers (ExceptionType, handler)
            where 'handler' is a callable that takes a `TestCase`, a
            ``testtools.TestResult`` and the exception raised.
        :param last_resort: Handler to call before re-raising uncatchable
            exceptions (those for which there is no handler).
        :param reactor: The Twisted reactor to use.  If not given, we use the
            default reactor.
        :param float timeout: The maximum time allowed for running a test.  The
            default is 0.005s.
        :param debug: Whether or not to enable Twisted's debugging.  Use this
            to get information about unhandled Deferreds and left-over
            DelayedCalls.  Defaults to False.
        :param bool suppress_twisted_logging: If True, then suppress Twisted's
            default logging while the test is being run. Defaults to True.
        :param bool store_twisted_logs: If True, then store the Twisted logs
            that took place during the run as the 'twisted-log' detail.
            Defaults to True.
        """
        super().__init__(case, handlers, last_resort)
        if reactor is None:
            from twisted.internet import reactor
        self._reactor = reactor
        self._timeout = timeout
        self._debug = debug
        self._suppress_twisted_logging = suppress_twisted_logging
        self._store_twisted_logs = store_twisted_logs

    @classmethod
    def make_factory(cls, reactor=None, timeout=0.005, debug=False, suppress_twisted_logging=True, store_twisted_logs=True):
        """Make a factory that conforms to the RunTest factory interface.

        Example::

            class SomeTests(TestCase):
                # Timeout tests after two minutes.
                run_tests_with = AsynchronousDeferredRunTest.make_factory(
                    timeout=120)
        """

        class AsynchronousDeferredRunTestFactory:

            def __call__(self, case, handlers=None, last_resort=None):
                return cls(case, handlers, last_resort, reactor, timeout, debug, suppress_twisted_logging, store_twisted_logs)
        return AsynchronousDeferredRunTestFactory()

    @defer.inlineCallbacks
    def _run_cleanups(self):
        """Run the cleanups on the test case.

        We expect that the cleanups on the test case can also return
        asynchronous Deferreds.  As such, we take the responsibility for
        running the cleanups, rather than letting TestCase do it.
        """
        last_exception = None
        while self.case._cleanups:
            f, args, kwargs = self.case._cleanups.pop()
            d = defer.maybeDeferred(f, *args, **kwargs)
            try:
                yield d
            except Exception:
                exc_info = sys.exc_info()
                self.case._report_traceback(exc_info)
                last_exception = exc_info[1]
        defer.returnValue(last_exception)

    def _make_spinner(self):
        """Make the `Spinner` to be used to run the tests."""
        return Spinner(self._reactor, debug=self._debug)

    def _run_deferred(self):
        """Run the test, assuming everything in it is Deferred-returning.

        This should return a Deferred that fires with True if the test was
        successful and False if the test was not successful.  It should *not*
        call addSuccess on the result, because there's reactor clean up that
        we needs to be done afterwards.
        """
        fails = []

        def fail_if_exception_caught(exception_caught):
            if self.exception_caught == exception_caught:
                fails.append(None)

        def clean_up(ignored=None):
            """Run the cleanups."""
            d = self._run_cleanups()

            def clean_up_done(result):
                if result is not None:
                    self._exceptions.append(result)
                    fails.append(None)
            return d.addCallback(clean_up_done)

        def set_up_done(exception_caught):
            """Set up is done, either clean up or run the test."""
            if self.exception_caught == exception_caught:
                fails.append(None)
                return clean_up()
            else:
                d = self._run_user(self.case._run_test_method, self.result)
                d.addCallback(fail_if_exception_caught)
                d.addBoth(tear_down)
                return d

        def tear_down(ignored):
            d = self._run_user(self.case._run_teardown, self.result)
            d.addCallback(fail_if_exception_caught)
            d.addBoth(clean_up)
            return d

        def force_failure(ignored):
            if getattr(self.case, 'force_failure', None):
                d = self._run_user(_raise_force_fail_error)
                d.addCallback(fails.append)
                return d
        d = self._run_user(self.case._run_setup, self.result)
        d.addCallback(set_up_done)
        d.addBoth(force_failure)
        d.addBoth(lambda ignored: len(fails) == 0)
        return d

    def _log_user_exception(self, e):
        """Raise 'e' and report it as a user exception."""
        try:
            raise e
        except e.__class__:
            self._got_user_exception(sys.exc_info())

    def _blocking_run_deferred(self, spinner):
        try:
            return trap_unhandled_errors(spinner.run, self._timeout, self._run_deferred)
        except NoResultError:
            self._got_user_exception(sys.exc_info())
            self.result.stop()
            return (False, [])
        except TimeoutError:
            self._log_user_exception(TimeoutError(self.case, self._timeout))
            return (False, [])

    def _get_log_fixture(self):
        """Return the log fixture we're configured to use."""
        fixtures = []
        if self._suppress_twisted_logging:
            fixtures.append(_NoTwistedLogObservers())
        if self._store_twisted_logs:
            fixtures.append(CaptureTwistedLogs())
        return CompoundFixture(fixtures)

    def _run_core(self):
        self.case.reactor = self._reactor
        spinner = self._make_spinner()
        with self._get_log_fixture() as capture_logs:
            for name, detail in capture_logs.getDetails().items():
                self.case.addDetail(name, detail)
            with _ErrorObserver(_log_observer) as error_fixture:
                successful, unhandled = self._blocking_run_deferred(spinner)
            for logged_error in error_fixture.flush_logged_errors():
                successful = False
                self._got_user_failure(logged_error, tb_label='logged-error')
        if unhandled:
            successful = False
            for debug_info in unhandled:
                f = debug_info.failResult
                info = debug_info._getDebugTracebacks()
                if info:
                    self.case.addDetail('unhandled-error-in-deferred-debug', text_content(info))
                self._got_user_failure(f, 'unhandled-error-in-deferred')
        junk = spinner.clear_junk()
        if junk:
            successful = False
            self._log_user_exception(UncleanReactorError(junk))
        if successful:
            self.result.addSuccess(self.case, details=self.case.getDetails())

    def _run_user(self, function, *args):
        """Run a user-supplied function.

        This just makes sure that it returns a Deferred, regardless of how the
        user wrote it.
        """
        d = defer.maybeDeferred(function, *args)
        return d.addErrback(self._got_user_failure)