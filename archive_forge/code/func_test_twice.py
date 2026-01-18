import csv
import datetime
import sys
import unittest
from io import StringIO
import testtools
from testtools import TestCase
from testtools.content import TracebackContent, text_content
from testtools.testresult.doubles import ExtendedTestResult
import subunit
import iso8601
import subunit.test_results
def test_twice(self):
    self.result.startTest(self)
    self.result.addSuccess(self, details={'foo': 'bar'})
    self.result.stopTest(self)
    self.result.startTest(self)
    self.result.addSuccess(self)
    self.result.stopTest(self)
    self.assertEqual([{'test': self, 'status': 'success', 'start_time': 0, 'stop_time': 1, 'tags': set(), 'details': {'foo': 'bar'}}, {'test': self, 'status': 'success', 'start_time': 2, 'stop_time': 3, 'tags': set(), 'details': None}], self.log)