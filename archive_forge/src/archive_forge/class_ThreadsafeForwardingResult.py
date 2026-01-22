import datetime
import email.message
import math
from operator import methodcaller
import sys
import unittest
import warnings
from testtools.compat import _b
from testtools.content import (
from testtools.content_type import ContentType
from testtools.tags import TagContext
class ThreadsafeForwardingResult(TestResult):
    """A TestResult which ensures the target does not receive mixed up calls.

    Multiple ``ThreadsafeForwardingResults`` can forward to the same target
    result, and that target result will only ever receive the complete set of
    events for one test at a time.

    This is enforced using a semaphore, which further guarantees that tests
    will be sent atomically even if the ``ThreadsafeForwardingResults`` are in
    different threads.

    ``ThreadsafeForwardingResult`` is typically used by
    ``ConcurrentTestSuite``, which creates one ``ThreadsafeForwardingResult``
    per thread, each of which wraps of the TestResult that
    ``ConcurrentTestSuite.run()`` is called with.

    target.startTestRun() and target.stopTestRun() are called once for each
    ThreadsafeForwardingResult that forwards to the same target. If the target
    takes special action on these events, it should take care to accommodate
    this.

    time() and tags() calls are batched to be adjacent to the test result and
    in the case of tags() are coerced into test-local scope, avoiding the
    opportunity for bugs around global state in the target.
    """

    def __init__(self, target, semaphore):
        """Create a ThreadsafeForwardingResult forwarding to target.

        :param target: A ``TestResult``.
        :param semaphore: A ``threading.Semaphore`` with limit 1.
        """
        TestResult.__init__(self)
        self.result = ExtendedToOriginalDecorator(target)
        self.semaphore = semaphore
        self._test_start = None
        self._global_tags = (set(), set())
        self._test_tags = (set(), set())

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.result!r}>'

    def _any_tags(self, tags):
        return bool(tags[0] or tags[1])

    def _add_result_with_semaphore(self, method, test, *args, **kwargs):
        now = self._now()
        self.semaphore.acquire()
        try:
            self.result.time(self._test_start)
            self.result.startTest(test)
            self.result.time(now)
            if self._any_tags(self._global_tags):
                self.result.tags(*self._global_tags)
            if self._any_tags(self._test_tags):
                self.result.tags(*self._test_tags)
            self._test_tags = (set(), set())
            try:
                method(test, *args, **kwargs)
            finally:
                self.result.stopTest(test)
        finally:
            self.semaphore.release()
        self._test_start = None

    def addError(self, test, err=None, details=None):
        self._add_result_with_semaphore(self.result.addError, test, err, details=details)

    def addExpectedFailure(self, test, err=None, details=None):
        self._add_result_with_semaphore(self.result.addExpectedFailure, test, err, details=details)

    def addFailure(self, test, err=None, details=None):
        self._add_result_with_semaphore(self.result.addFailure, test, err, details=details)

    def addSkip(self, test, reason=None, details=None):
        self._add_result_with_semaphore(self.result.addSkip, test, reason, details=details)

    def addSuccess(self, test, details=None):
        self._add_result_with_semaphore(self.result.addSuccess, test, details=details)

    def addUnexpectedSuccess(self, test, details=None):
        self._add_result_with_semaphore(self.result.addUnexpectedSuccess, test, details=details)

    def progress(self, offset, whence):
        pass

    def startTestRun(self):
        super().startTestRun()
        self.semaphore.acquire()
        try:
            self.result.startTestRun()
        finally:
            self.semaphore.release()

    def _get_shouldStop(self):
        self.semaphore.acquire()
        try:
            return self.result.shouldStop
        finally:
            self.semaphore.release()

    def _set_shouldStop(self, value):
        pass
    shouldStop = property(_get_shouldStop, _set_shouldStop)

    def stop(self):
        self.semaphore.acquire()
        try:
            self.result.stop()
        finally:
            self.semaphore.release()

    def stopTestRun(self):
        self.semaphore.acquire()
        try:
            self.result.stopTestRun()
        finally:
            self.semaphore.release()

    def done(self):
        self.semaphore.acquire()
        try:
            self.result.done()
        finally:
            self.semaphore.release()

    def startTest(self, test):
        self._test_start = self._now()
        super().startTest(test)

    def wasSuccessful(self):
        return self.result.wasSuccessful()

    def tags(self, new_tags, gone_tags):
        """See `TestResult`."""
        super().tags(new_tags, gone_tags)
        if self._test_start is not None:
            self._test_tags = _merge_tags(self._test_tags, (new_tags, gone_tags))
        else:
            self._global_tags = _merge_tags(self._global_tags, (new_tags, gone_tags))