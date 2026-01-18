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
def test_skipException_in_setup_calls_result_addSkip(self):

    class TestThatRaisesInSetUp(TestCase):

        def setUp(self):
            TestCase.setUp(self)
            self.skipTest('skipping this test')

        def test_that_passes(self):
            pass
    calls = []
    result = LoggingResult(calls)
    test = TestThatRaisesInSetUp('test_that_passes')
    test.run(result)
    case = result._events[0][1]
    self.assertEqual([('startTest', case), ('addSkip', case, 'skipping this test'), ('stopTest', case)], calls)