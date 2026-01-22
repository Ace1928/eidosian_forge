import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
def _end_test(self, test_id):
    test_start = self._active_tests.pop(test_id, None)
    if not test_start:
        test_duration = 0
    else:
        test_duration = self._time() - test_start
    self.reportTest(test_id, test_duration)