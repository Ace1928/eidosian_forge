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
class StreamSummary(StreamResult):
    """A specialised StreamResult that summarises a stream.

    The summary uses the same representation as the original
    unittest.TestResult contract, allowing it to be consumed by any test
    runner.
    """

    def __init__(self):
        super().__init__()
        self._hook = _StreamToTestRecord(self._gather_test)
        self._handle_status = {'success': self._success, 'skip': self._skip, 'exists': self._exists, 'fail': self._fail, 'xfail': self._xfail, 'uxsuccess': self._uxsuccess, 'unknown': self._incomplete, 'inprogress': self._incomplete}

    def startTestRun(self):
        super().startTestRun()
        self.failures = []
        self.errors = []
        self.testsRun = 0
        self.skipped = []
        self.expectedFailures = []
        self.unexpectedSuccesses = []
        self._hook.startTestRun()

    def status(self, *args, **kwargs):
        super().status(*args, **kwargs)
        self._hook.status(*args, **kwargs)

    def stopTestRun(self):
        super().stopTestRun()
        self._hook.stopTestRun()

    def wasSuccessful(self):
        """Return False if any failure has occurred.

        Note that incomplete tests can only be detected when stopTestRun is
        called, so that should be called before checking wasSuccessful.
        """
        return not self.failures and (not self.errors)

    def _gather_test(self, test_record):
        if test_record.status == 'exists':
            return
        self.testsRun += 1
        case = test_record.to_test_case()
        self._handle_status[test_record.status](case)

    def _incomplete(self, case):
        self.errors.append((case, 'Test did not complete'))

    def _success(self, case):
        pass

    def _skip(self, case):
        if 'reason' not in case._details:
            reason = 'Unknown'
        else:
            reason = case._details['reason'].as_text()
        self.skipped.append((case, reason))

    def _exists(self, case):
        pass

    def _fail(self, case):
        message = _details_to_str(case._details, special='traceback')
        self.errors.append((case, message))

    def _xfail(self, case):
        message = _details_to_str(case._details, special='traceback')
        self.expectedFailures.append((case, message))

    def _uxsuccess(self, case):
        case._outcome = 'addUnexpectedSuccess'
        self.unexpectedSuccesses.append(case)