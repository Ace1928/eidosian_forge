import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
class TestIdPrintingResult(testtools.TestResult):
    """Print test ids to a stream.

    Implements both TestResult and StreamResult, for compatibility.
    """

    def __init__(self, stream, show_times=False, show_exists=False):
        """Create a FilterResult object outputting to stream."""
        super().__init__()
        self._stream = stream
        self.show_exists = show_exists
        self.show_times = show_times

    def startTestRun(self):
        self.failed_tests = 0
        self.__time = None
        self._test = None
        self._test_duration = 0
        self._active_tests = {}

    def addError(self, test, err):
        self.failed_tests += 1
        self._test = test

    def addFailure(self, test, err):
        self.failed_tests += 1
        self._test = test

    def addSuccess(self, test):
        self._test = test

    def addSkip(self, test, reason=None, details=None):
        self._test = test

    def addUnexpectedSuccess(self, test, details=None):
        self.failed_tests += 1
        self._test = test

    def addExpectedFailure(self, test, err=None, details=None):
        self._test = test

    def reportTest(self, test_id, duration):
        if self.show_times:
            seconds = duration.seconds
            seconds += duration.days * 3600 * 24
            seconds += duration.microseconds / 1000000.0
            self._stream.write(test_id + ' %0.3f\n' % seconds)
        else:
            self._stream.write(test_id + '\n')

    def startTest(self, test):
        self._start_time = self._time()

    def status(self, test_id=None, test_status=None, test_tags=None, runnable=True, file_name=None, file_bytes=None, eof=False, mime_type=None, route_code=None, timestamp=None):
        if not test_id:
            return
        if timestamp is not None:
            self.time(timestamp)
        if test_status == 'exists':
            if self.show_exists:
                self.reportTest(test_id, 0)
        elif test_status in ('inprogress', None):
            self._active_tests[test_id] = self._time()
        else:
            self._end_test(test_id)

    def _end_test(self, test_id):
        test_start = self._active_tests.pop(test_id, None)
        if not test_start:
            test_duration = 0
        else:
            test_duration = self._time() - test_start
        self.reportTest(test_id, test_duration)

    def stopTest(self, test):
        test_duration = self._time() - self._start_time
        self.reportTest(self._test.id(), test_duration)

    def time(self, time):
        self.__time = time

    def _time(self):
        return self.__time

    def wasSuccessful(self):
        """Tells whether or not this result was a success"""
        return self.failed_tests == 0

    def stopTestRun(self):
        for test_id in list(self._active_tests.keys()):
            self._end_test(test_id)