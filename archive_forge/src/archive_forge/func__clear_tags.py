import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
def _clear_tags(self):
    self._global_tags = (set(), set())
    self._test_tags = None