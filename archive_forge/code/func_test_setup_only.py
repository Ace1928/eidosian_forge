import types
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
def test_setup_only(self):

    class Stub:

        def setUp(self):
            self.value = 42
    fixture = fixtures.MethodFixture(Stub())
    fixture.setUp()
    self.assertEqual(42, fixture.obj.value)
    self.assertIsInstance(fixture.obj, Stub)
    fixture.cleanUp()