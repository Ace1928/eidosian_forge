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
def test_without_time_calls_time_is_called_and_not_None(self):
    self.result.startTest(self)
    self.assertEqual(1, len(self.decorated._calls))
    self.assertNotEqual(None, self.decorated._calls[0])