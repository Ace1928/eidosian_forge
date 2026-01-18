import types
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
def test_setup_failures_with_base_exception(self):

    class MyBase(BaseException):
        pass
    log = []

    class Subclass(fixtures.Fixture):

        def _setUp(self):
            self.addDetail('log', text_content('stuff'))
            self.addCleanup(log.append, 'cleaned')
            raise MyBase('fred')
    f = Subclass()
    self.assertRaises(MyBase, f.setUp)
    self.assertRaises(TypeError, f.cleanUp)
    self.assertEqual(['cleaned'], log)