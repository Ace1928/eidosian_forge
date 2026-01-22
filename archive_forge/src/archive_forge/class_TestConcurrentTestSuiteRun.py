import doctest
from pprint import pformat
import unittest
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import DocTestMatches, Equals
from testtools.testresult.doubles import StreamResult as LoggingStream
from testtools.testsuite import FixtureSuite, sorted_tests
from testtools.tests.helpers import LoggingResult
class TestConcurrentTestSuiteRun(TestCase):

    def test_broken_test(self):
        log = []

        def on_test(test, status, start_time, stop_time, tags, details):
            log.append((test.id(), status, set(details.keys())))

        class BrokenTest:

            def __call__(self):
                pass
            run = __call__
        original_suite = unittest.TestSuite([BrokenTest()])
        suite = ConcurrentTestSuite(original_suite, self.split_suite)
        suite.run(TestByTestResult(on_test))
        self.assertEqual([('broken-runner', 'error', {'traceback'})], log)

    def test_trivial(self):
        log = []
        result = LoggingResult(log)
        test1 = Sample('test_method1')
        test2 = Sample('test_method2')
        original_suite = unittest.TestSuite([test1, test2])
        suite = ConcurrentTestSuite(original_suite, self.split_suite)
        suite.run(result)
        test1 = log[1][1]
        test2 = log[-1][1]
        self.assertIsInstance(test1, Sample)
        self.assertIsInstance(test2, Sample)
        self.assertNotEqual(test1.id(), test2.id())

    def test_wrap_result(self):
        wrap_log = []

        def wrap_result(thread_safe_result, thread_number):
            wrap_log.append((thread_safe_result.result.decorated, thread_number))
            return thread_safe_result
        result_log = []
        result = LoggingResult(result_log)
        test1 = Sample('test_method1')
        test2 = Sample('test_method2')
        original_suite = unittest.TestSuite([test1, test2])
        suite = ConcurrentTestSuite(original_suite, self.split_suite, wrap_result=wrap_result)
        suite.run(result)
        self.assertEqual([(result, 0), (result, 1)], wrap_log)
        self.assertNotEqual([], result_log)

    def split_suite(self, suite):
        return list(iterate_tests(suite))