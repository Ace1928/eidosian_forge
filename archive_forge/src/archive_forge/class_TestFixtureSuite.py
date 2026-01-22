import doctest
from pprint import pformat
import unittest
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import DocTestMatches, Equals
from testtools.testresult.doubles import StreamResult as LoggingStream
from testtools.testsuite import FixtureSuite, sorted_tests
from testtools.tests.helpers import LoggingResult
class TestFixtureSuite(TestCase):

    def setUp(self):
        super().setUp()
        if FunctionFixture is None:
            self.skipTest('Need fixtures')

    def test_fixture_suite(self):
        log = []

        class Sample(TestCase):

            def test_one(self):
                log.append(1)

            def test_two(self):
                log.append(2)
        fixture = FunctionFixture(lambda: log.append('setUp'), lambda fixture: log.append('tearDown'))
        suite = FixtureSuite(fixture, [Sample('test_one'), Sample('test_two')])
        suite.run(LoggingResult([]))
        self.assertEqual(['setUp', 1, 2, 'tearDown'], log)

    def test_fixture_suite_sort(self):
        log = []

        class Sample(TestCase):

            def test_one(self):
                log.append(1)

            def test_two(self):
                log.append(2)
        fixture = FunctionFixture(lambda: log.append('setUp'), lambda fixture: log.append('tearDown'))
        suite = FixtureSuite(fixture, [Sample('test_one'), Sample('test_one')])
        self.assertRaises(ValueError, suite.sort_tests)