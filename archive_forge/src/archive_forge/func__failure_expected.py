import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
def _failure_expected(self, test):
    return test.id() in self._fixup_expected_failures