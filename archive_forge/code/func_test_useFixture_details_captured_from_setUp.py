import unittest
from testtools import (
from testtools.compat import _b
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.testresult.doubles import (
def test_useFixture_details_captured_from_setUp(self):

    class BrokenFixture(fixtures.Fixture):

        def setUp(self):
            fixtures.Fixture.setUp(self)
            self.addDetail('content', content.text_content('foobar'))
            raise Exception()
    fixture = BrokenFixture()

    class SimpleTest(TestCase):

        def test_foo(self):
            self.useFixture(fixture)
    result = ExtendedTestResult()
    SimpleTest('test_foo').run(result)
    self.assertEqual('addError', result._events[-2][0])
    details = result._events[-2][2]
    self.assertEqual(['content', 'traceback'], sorted(details))
    self.assertEqual('foobar', ''.join(details['content'].iter_text()))