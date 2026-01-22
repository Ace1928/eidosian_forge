import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
class TestByTestResult(testtools.TestResult):
    """Call something every time a test completes."""

    def __init__(self, on_test):
        """Construct a ``TestByTestResult``.

        :param on_test: A callable that take a test case, a status (one of
            "success", "failure", "error", "skip", or "xfail"), a start time
            (a ``datetime`` with timezone), a stop time, an iterable of tags,
            and a details dict. Is called at the end of each test (i.e. on
            ``stopTest``) with the accumulated values for that test.
        """
        super().__init__()
        self._on_test = on_test

    def startTest(self, test):
        super().startTest(test)
        self._start_time = self._now()
        self._status = None
        self._details = None
        self._stop_time = None

    def stopTest(self, test):
        self._stop_time = self._now()
        super().stopTest(test)
        self._on_test(test=test, status=self._status, start_time=self._start_time, stop_time=self._stop_time, tags=getattr(self, 'current_tags', None), details=self._details)

    def _err_to_details(self, test, err, details):
        if details:
            return details
        return {'traceback': TracebackContent(err, test)}

    def addSuccess(self, test, details=None):
        super().addSuccess(test)
        self._status = 'success'
        self._details = details

    def addFailure(self, test, err=None, details=None):
        super().addFailure(test, err, details)
        self._status = 'failure'
        self._details = self._err_to_details(test, err, details)

    def addError(self, test, err=None, details=None):
        super().addError(test, err, details)
        self._status = 'error'
        self._details = self._err_to_details(test, err, details)

    def addSkip(self, test, reason=None, details=None):
        super().addSkip(test, reason, details)
        self._status = 'skip'
        if details is None:
            details = {'reason': text_content(reason)}
        elif reason:
            details['reason'] = text_content(reason)
        self._details = details

    def addExpectedFailure(self, test, err=None, details=None):
        super().addExpectedFailure(test, err, details)
        self._status = 'xfail'
        self._details = self._err_to_details(test, err, details)

    def addUnexpectedSuccess(self, test, details=None):
        super().addUnexpectedSuccess(test, details)
        self._status = 'success'
        self._details = details