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
def test_tags_collapsed_outside_of_tests(self):
    result = ExtendedTestResult()
    tag_collapser = subunit.test_results.TagCollapsingDecorator(result)
    tag_collapser.tags({'a'}, set())
    tag_collapser.tags({'b'}, set())
    tag_collapser.startTest(self)
    self.assertEqual([('tags', {'a', 'b'}, set()), ('startTest', self)], result._events)