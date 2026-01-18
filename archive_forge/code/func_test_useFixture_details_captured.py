import unittest
from testtools import (
from testtools.compat import _b
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.testresult.doubles import (
def test_useFixture_details_captured(self):

    class DetailsFixture(fixtures.Fixture):

        def setUp(self):
            fixtures.Fixture.setUp(self)
            self.addCleanup(delattr, self, 'content')
            self.content = [_b('content available until cleanUp')]
            self.addDetail('content', content.Content(content_type.UTF8_TEXT, self.get_content))

        def get_content(self):
            return self.content
    fixture = DetailsFixture()

    class SimpleTest(TestCase):

        def test_foo(self):
            self.useFixture(fixture)
            self.addDetail('content', content.Content(content_type.UTF8_TEXT, lambda: [_b('foo')]))
    result = ExtendedTestResult()
    SimpleTest('test_foo').run(result)
    self.assertEqual('addSuccess', result._events[-2][0])
    details = result._events[-2][2]
    self.assertEqual(['content', 'content-1'], sorted(details.keys()))
    self.assertEqual('foo', details['content'].as_text())
    self.assertEqual('content available until cleanUp', details['content-1'].as_text())