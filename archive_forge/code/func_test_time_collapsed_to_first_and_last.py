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
def test_time_collapsed_to_first_and_last(self):
    result = ExtendedTestResult()
    tag_collapser = subunit.test_results.TimeCollapsingDecorator(result)
    times = [self.make_time() for i in range(5)]
    for a_time in times:
        tag_collapser.time(a_time)
    tag_collapser.startTest(subunit.RemotedTestCase('foo'))
    self.assertEqual([('time', times[0]), ('time', times[-1])], result._events[:-1])