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
def assertDetailsProvided(self, case, expected_outcome, expected_keys):
    """Assert that when case is run, details are provided to the result.

        :param case: A TestCase to run.
        :param expected_outcome: The call that should be made.
        :param expected_keys: The keys to look for.
        """
    result = ExtendedTestResult()
    case.run(result)
    case = result._events[0][1]
    expected = [('startTest', case), (expected_outcome, case), ('stopTest', case)]
    self.assertEqual(3, len(result._events))
    self.assertEqual(expected[0], result._events[0])
    self.assertEqual(expected[1], result._events[1][0:2])
    self.assertEqual(sorted(expected_keys), sorted(result._events[1][2].keys()))
    self.assertEqual(expected[-1], result._events[-1])