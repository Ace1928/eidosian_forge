import unittest
from testtools import (
from testtools.compat import _b
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.testresult.doubles import (
def test_useFixture_details_captured_from__setUp(self):

    class BrokenFixture(fixtures.Fixture):

        def _setUp(self):
            fixtures.Fixture._setUp(self)
            self.addDetail('broken', content.text_content('foobar'))
            raise Exception('_setUp broke')
    fixture = BrokenFixture()

    class SimpleTest(TestCase):

        def test_foo(self):
            self.addDetail('foo_content', content.text_content('foo ok'))
            self.useFixture(fixture)
    result = ExtendedTestResult()
    SimpleTest('test_foo').run(result)
    self.assertEqual('addError', result._events[-2][0])
    details = result._events[-2][2]
    self.assertEqual(['broken', 'foo_content', 'traceback', 'traceback-1'], sorted(details))
    self.expectThat(''.join(details['broken'].iter_text()), Equals('foobar'))
    self.expectThat(''.join(details['foo_content'].iter_text()), Equals('foo ok'))
    self.expectThat(''.join(details['traceback'].iter_text()), Contains('_setUp broke'))
    self.expectThat(''.join(details['traceback-1'].iter_text()), Contains('foobar'))