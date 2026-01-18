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
def test_skip_preserved(self):
    subunit_stream = _b('\n'.join(['test: foo', 'skip: foo', '']))
    result = ExtendedTestResult()
    result_filter = TestResultFilter(result)
    self.run_tests(result_filter, subunit_stream)
    foo = subunit.RemotedTestCase('foo')
    self.assertEqual([('startTest', foo), ('addSkip', foo, {}), ('stopTest', foo)], result._events)