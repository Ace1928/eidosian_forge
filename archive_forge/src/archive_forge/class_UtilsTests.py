import testtools
from keystoneauth1 import _utils
class UtilsTests(testtools.TestCase):

    def test_get_logger(self):
        self.assertEqual('keystoneauth.tests.unit.test_utils', _utils.get_logger(__name__).name)