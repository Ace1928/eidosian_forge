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
def test_skip_without_reason_works(self):

    class Test(TestCase):

        def test(self):
            raise self.skipException()
    case = Test('test')
    result = ExtendedTestResult()
    case.run(result)
    self.assertEqual('addSkip', result._events[1][0])
    self.assertEqual('no reason given.', result._events[1][2]['reason'].as_text())