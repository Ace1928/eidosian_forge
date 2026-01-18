import doctest
from pprint import pformat
import unittest
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import DocTestMatches, Equals
from testtools.testresult.doubles import StreamResult as LoggingStream
from testtools.testsuite import FixtureSuite, sorted_tests
from testtools.tests.helpers import LoggingResult
def test_sorts_custom_suites(self):
    a = PlaceHolder('a')
    b = PlaceHolder('b')

    class Subclass(unittest.TestSuite):

        def sort_tests(self):
            self._tests = sorted_tests(self, True)
    input_suite = Subclass([b, a])
    suite = sorted_tests(input_suite)
    self.assertEqual([a, b], list(iterate_tests(suite)))
    self.assertEqual([input_suite], list(iter(suite)))