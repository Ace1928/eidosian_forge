import unittest
from testtools import (
from testtools.compat import _b
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.testresult.doubles import (
def test_useFixture(self):
    fixture = LoggingFixture()

    class SimpleTest(TestCase):

        def test_foo(self):
            self.useFixture(fixture)
    result = unittest.TestResult()
    SimpleTest('test_foo').run(result)
    self.assertTrue(result.wasSuccessful())
    self.assertEqual(['setUp', 'cleanUp'], fixture.calls)