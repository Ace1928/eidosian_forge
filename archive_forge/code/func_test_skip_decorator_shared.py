from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
def test_skip_decorator_shared(self):

    def shared(testcase):
        testcase.fail('nope')

    class SkippingTest(TestCase):
        test_skip = skipIf(True, 'skipping this test')(shared)

    class NotSkippingTest(TestCase):
        test_no_skip = skipIf(False, 'skipping this test')(shared)
    events = []
    result = Python26TestResult(events)
    test = SkippingTest('test_skip')
    test.run(result)
    self.assertEqual('addSuccess', events[1][0])
    events2 = []
    result2 = Python26TestResult(events2)
    test2 = NotSkippingTest('test_no_skip')
    test2.run(result2)
    self.assertEqual('addFailure', events2[1][0])