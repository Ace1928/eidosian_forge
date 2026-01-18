import types
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
@require_gather_details
def test_useFixture_details_captured_from_setUp_MultipleExceptions(self):

    class SomethingBroke(Exception):
        pass

    class BrokenFixture(fixtures.Fixture):

        def _setUp(self):
            self.addDetail('content', text_content('foobar'))
            raise SomethingBroke()

    class SimpleFixture(fixtures.Fixture):

        def _setUp(self):
            self.useFixture(BrokenFixture())
    simple = SimpleFixture()
    e = self.assertRaises(fixtures.MultipleExceptions, simple.setUp)
    self.assertEqual({'content': text_content('foobar')}, e.args[-1][1].args[0])