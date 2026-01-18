import unittest
from testtools import (
from testtools.compat import _b
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.testresult.doubles import (
def test_useFixture_multiple_details_captured(self):

    class DetailsFixture(fixtures.Fixture):

        def setUp(self):
            fixtures.Fixture.setUp(self)
            self.addDetail('aaa', content.text_content('foo'))
            self.addDetail('bbb', content.text_content('bar'))
    fixture = DetailsFixture()

    class SimpleTest(TestCase):

        def test_foo(self):
            self.useFixture(fixture)
    result = ExtendedTestResult()
    SimpleTest('test_foo').run(result)
    self.assertEqual('addSuccess', result._events[-2][0])
    details = result._events[-2][2]
    self.assertEqual(['aaa', 'bbb'], sorted(details))
    self.assertEqual('foo', details['aaa'].as_text())
    self.assertEqual('bar', details['bbb'].as_text())