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
class ExtendedToOriginalDecorator:
    """Permit new TestResult API code to degrade gracefully with old results.

    This decorates an existing TestResult and converts missing outcomes
    such as addSkip to older outcomes such as addSuccess. It also supports
    the extended details protocol. In all cases the most recent protocol
    is attempted first, and fallbacks only occur when the decorated result
    does not support the newer style of calling.
    """

    def __init__(self, decorated):
        self.decorated = decorated
        self._tags = TagContext()
        self._failfast = False
        self._shouldStop = False

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.decorated!r}>'

    def __getattr__(self, name):
        return getattr(self.decorated, name)

    def addError(self, test, err=None, details=None):
        try:
            self._check_args(err, details)
            if details is not None:
                try:
                    return self.decorated.addError(test, details=details)
                except TypeError:
                    err = self._details_to_exc_info(details)
            return self.decorated.addError(test, err)
        finally:
            if self.failfast:
                self.stop()

    def addExpectedFailure(self, test, err=None, details=None):
        self._check_args(err, details)
        addExpectedFailure = getattr(self.decorated, 'addExpectedFailure', None)
        if addExpectedFailure is None:
            return self.addSuccess(test)
        if details is not None:
            try:
                return addExpectedFailure(test, details=details)
            except TypeError:
                err = self._details_to_exc_info(details)
        return addExpectedFailure(test, err)

    def addFailure(self, test, err=None, details=None):
        try:
            self._check_args(err, details)
            if details is not None:
                try:
                    return self.decorated.addFailure(test, details=details)
                except TypeError:
                    err = self._details_to_exc_info(details)
            return self.decorated.addFailure(test, err)
        finally:
            if self.failfast:
                self.stop()

    def addSkip(self, test, reason=None, details=None):
        self._check_args(reason, details)
        addSkip = getattr(self.decorated, 'addSkip', None)
        if addSkip is None:
            return self.decorated.addSuccess(test)
        if details is not None:
            try:
                return addSkip(test, details=details)
            except TypeError:
                try:
                    reason = details['reason'].as_text()
                except KeyError:
                    reason = _details_to_str(details)
        return addSkip(test, reason)

    def addUnexpectedSuccess(self, test, details=None):
        try:
            outcome = getattr(self.decorated, 'addUnexpectedSuccess', None)
            if outcome is None:
                try:
                    test.fail('')
                except test.failureException:
                    return self.addFailure(test, sys.exc_info())
            if details is not None:
                try:
                    return outcome(test, details=details)
                except TypeError:
                    pass
            return outcome(test)
        finally:
            if self.failfast:
                self.stop()

    def addSuccess(self, test, details=None):
        if details is not None:
            try:
                return self.decorated.addSuccess(test, details=details)
            except TypeError:
                pass
        return self.decorated.addSuccess(test)

    def _check_args(self, err, details):
        param_count = 0
        if err is not None:
            param_count += 1
        if details is not None:
            param_count += 1
        if param_count != 1:
            raise ValueError("Must pass only one of err '%s' and details '%s" % (err, details))

    def _details_to_exc_info(self, details):
        """Convert a details dict to an exc_info tuple."""
        return (_StringException, _StringException(_details_to_str(details, special='traceback')), None)

    @property
    def current_tags(self):
        return getattr(self.decorated, 'current_tags', self._tags.get_current_tags())

    def done(self):
        try:
            return self.decorated.done()
        except AttributeError:
            return

    def _get_failfast(self):
        return getattr(self.decorated, 'failfast', self._failfast)

    def _set_failfast(self, value):
        if hasattr(self.decorated, 'failfast'):
            self.decorated.failfast = value
        else:
            self._failfast = value
    failfast = property(_get_failfast, _set_failfast)

    def progress(self, offset, whence):
        method = getattr(self.decorated, 'progress', None)
        if method is None:
            return
        return method(offset, whence)

    def _get_shouldStop(self):
        return getattr(self.decorated, 'shouldStop', self._shouldStop)

    def _set_shouldStop(self, value):
        if hasattr(self.decorated, 'shouldStop'):
            self.decorated.shouldStop = value
        else:
            self._shouldStop = value
    shouldStop = property(_get_shouldStop, _set_shouldStop)

    def startTest(self, test):
        self._tags = TagContext(self._tags)
        return self.decorated.startTest(test)

    def startTestRun(self):
        self._tags = TagContext()
        try:
            return self.decorated.startTestRun()
        except AttributeError:
            return

    def stop(self):
        method = getattr(self.decorated, 'stop', None)
        if method:
            return method()
        self.shouldStop = True

    def stopTest(self, test):
        self._tags = self._tags.parent
        return self.decorated.stopTest(test)

    def stopTestRun(self):
        try:
            return self.decorated.stopTestRun()
        except AttributeError:
            return

    def tags(self, new_tags, gone_tags):
        method = getattr(self.decorated, 'tags', None)
        if method is not None:
            return method(new_tags, gone_tags)
        else:
            self._tags.change_tags(new_tags, gone_tags)

    def time(self, a_datetime):
        method = getattr(self.decorated, 'time', None)
        if method is None:
            return
        return method(a_datetime)

    def wasSuccessful(self):
        return self.decorated.wasSuccessful()