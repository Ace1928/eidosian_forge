import unittest
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures import TestWithFixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
class TestTestWithFixtures(unittest.TestCase):

    def test_useFixture(self):
        fixture = LoggingFixture()

        class SimpleTest(testtools.TestCase, TestWithFixtures):

            def test_foo(self):
                self.useFixture(fixture)
        result = unittest.TestResult()
        SimpleTest('test_foo').run(result)
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(['setUp', 'cleanUp'], fixture.calls)

    def test_useFixture_uses_raise_first(self):
        calls = []

        def raiser(ignored):
            calls.append('called')
            raise Exception('foo')
        fixture = fixtures.FunctionFixture(lambda: None, raiser)

        class SimpleTest(testtools.TestCase, TestWithFixtures):

            def test_foo(self):
                self.useFixture(fixture)
        result = unittest.TestResult()
        SimpleTest('test_foo').run(result)
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(['called'], calls)

    @skipIf(gather_details is None, 'gather_details() is not available.')
    def test_useFixture_details_captured_from_setUp(self):

        class SomethingBroke(Exception):
            pass

        class BrokenFixture(fixtures.Fixture):

            def setUp(self):
                super(BrokenFixture, self).setUp()
                self.addDetail('content', text_content('foobar'))
                raise SomethingBroke()
        broken_fixture = BrokenFixture()

        class DetailedTestCase(TestWithFixtures, testtools.TestCase):

            def setUp(self):
                super(DetailedTestCase, self).setUp()
                self.useFixture(broken_fixture)

            def test(self):
                pass
        detailed_test_case = DetailedTestCase('test')
        self.assertRaises(SomethingBroke, detailed_test_case.setUp)
        self.assertEqual({'content': text_content('foobar')}, broken_fixture.getDetails())
        self.assertEqual({'content': text_content('foobar')}, detailed_test_case.getDetails())

    @skipIf(gather_details is None, 'gather_details() is not available.')
    def test_useFixture_details_not_captured_from_setUp(self):

        class SomethingBroke(Exception):
            pass

        class BrokenFixture(fixtures.Fixture):

            def setUp(self):
                super(BrokenFixture, self).setUp()
                self.addDetail('content', text_content('foobar'))
                raise SomethingBroke()
        broken_fixture = BrokenFixture()

        class NonDetailedTestCase(TestWithFixtures, unittest.TestCase):

            def setUp(self):
                super(NonDetailedTestCase, self).setUp()
                self.useFixture(broken_fixture)

            def test(self):
                pass
        non_detailed_test_case = NonDetailedTestCase('test')
        self.assertRaises(SomethingBroke, non_detailed_test_case.setUp)