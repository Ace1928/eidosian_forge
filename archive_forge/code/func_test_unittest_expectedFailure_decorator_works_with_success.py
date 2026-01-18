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
def test_unittest_expectedFailure_decorator_works_with_success(self):

    class ReferenceTest(TestCase):

        @unittest.expectedFailure
        def test_passes_unexpectedly(self):
            self.assertEqual(1, 1)
    test = ReferenceTest('test_passes_unexpectedly')
    result = test.run()
    self.assertEqual(False, result.wasSuccessful())