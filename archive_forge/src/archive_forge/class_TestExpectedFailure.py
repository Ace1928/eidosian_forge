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
class TestExpectedFailure(TestWithDetails):
    """Tests for expected failures and unexpected successess."""
    run_test_with = FullStackRunTest

    def make_unexpected_case(self):

        class Case(TestCase):

            def test(self):
                raise testcase._UnexpectedSuccess
        case = Case('test')
        return case

    def test_raising__UnexpectedSuccess_py27(self):
        case = self.make_unexpected_case()
        result = Python27TestResult()
        case.run(result)
        case = result._events[0][1]
        self.assertEqual([('startTest', case), ('addUnexpectedSuccess', case), ('stopTest', case)], result._events)

    def test_raising__UnexpectedSuccess_extended(self):
        case = self.make_unexpected_case()
        result = ExtendedTestResult()
        case.run(result)
        case = result._events[0][1]
        self.assertEqual([('startTest', case), ('addUnexpectedSuccess', case, {}), ('stopTest', case)], result._events)

    def make_xfail_case_xfails(self):
        content = self.get_content()

        class Case(TestCase):

            def test(self):
                self.addDetail('foo', content)
                self.expectFailure('we are sad', self.assertEqual, 1, 0)
        case = Case('test')
        return case

    def make_xfail_case_succeeds(self):
        content = self.get_content()

        class Case(TestCase):

            def test(self):
                self.addDetail('foo', content)
                self.expectFailure('we are sad', self.assertEqual, 1, 1)
        case = Case('test')
        return case

    def test_expectFailure_KnownFailure_extended(self):
        case = self.make_xfail_case_xfails()
        self.assertDetailsProvided(case, 'addExpectedFailure', ['foo', 'traceback', 'reason'])

    def test_expectFailure_KnownFailure_unexpected_success(self):
        case = self.make_xfail_case_succeeds()
        self.assertDetailsProvided(case, 'addUnexpectedSuccess', ['foo', 'reason'])

    def test_unittest_expectedFailure_decorator_works_with_failure(self):

        class ReferenceTest(TestCase):

            @unittest.expectedFailure
            def test_fails_expectedly(self):
                self.assertEqual(1, 0)
        test = ReferenceTest('test_fails_expectedly')
        result = test.run()
        self.assertEqual(True, result.wasSuccessful())

    def test_unittest_expectedFailure_decorator_works_with_success(self):

        class ReferenceTest(TestCase):

            @unittest.expectedFailure
            def test_passes_unexpectedly(self):
                self.assertEqual(1, 1)
        test = ReferenceTest('test_passes_unexpectedly')
        result = test.run()
        self.assertEqual(False, result.wasSuccessful())