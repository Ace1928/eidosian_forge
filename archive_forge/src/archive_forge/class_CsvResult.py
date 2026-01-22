import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
class CsvResult(TestByTestResult):

    def __init__(self, stream):
        super().__init__(self._on_test)
        self._write_row = csv.writer(stream).writerow

    def _on_test(self, test, status, start_time, stop_time, tags, details):
        self._write_row([test.id(), status, start_time, stop_time])

    def startTestRun(self):
        super().startTestRun()
        self._write_row(['test', 'status', 'start_time', 'stop_time'])