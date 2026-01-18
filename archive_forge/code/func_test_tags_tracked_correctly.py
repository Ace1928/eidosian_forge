import subprocess
import sys
import unittest
from datetime import datetime
from io import BytesIO
from testtools import TestCase
from testtools.compat import _b
from testtools.testresult.doubles import ExtendedTestResult, StreamResult
import iso8601
import subunit
from subunit.test_results import make_tag_filter, TestResultFilter
from subunit import ByteStreamToStreamResult, StreamResultToBytes
def test_tags_tracked_correctly(self):
    tag_filter = make_tag_filter(['a'], [])
    result = ExtendedTestResult()
    result_filter = TestResultFilter(result, filter_success=False, filter_predicate=tag_filter)
    input_stream = _b('test: foo\ntags: a\nsuccessful: foo\ntest: bar\nsuccessful: bar\n')
    self.run_tests(result_filter, input_stream)
    foo = subunit.RemotedTestCase('foo')
    self.assertEqual([('startTest', foo), ('tags', {'a'}, set()), ('addSuccess', foo), ('stopTest', foo)], result._events)