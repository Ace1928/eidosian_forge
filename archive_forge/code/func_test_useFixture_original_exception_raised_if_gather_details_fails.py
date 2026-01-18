import unittest
from testtools import (
from testtools.compat import _b
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.testresult.doubles import (
def test_useFixture_original_exception_raised_if_gather_details_fails(self):

    class BrokenFixture(fixtures.Fixture):

        def getDetails(self):
            raise AttributeError('getDetails broke')

        def setUp(self):
            fixtures.Fixture.setUp(self)
            raise Exception('setUp broke')
    fixture = BrokenFixture()

    class SimpleTest(TestCase):

        def test_foo(self):
            self.useFixture(fixture)
    result = ExtendedTestResult()
    SimpleTest('test_foo').run(result)
    self.assertEqual('addError', result._events[-2][0])
    details = result._events[-2][2]
    self.assertEqual(['traceback', 'traceback-1'], sorted(details))
    self.assertThat(''.join(details['traceback'].iter_text()), Contains('setUp broke'))
    self.assertThat(''.join(details['traceback-1'].iter_text()), Contains('getDetails broke'))