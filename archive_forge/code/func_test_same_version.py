from oslotest import base as test_base
from oslo_utils import versionutils
def test_same_version(self):
    self.assertTrue(versionutils.is_compatible('1', '1'))
    self.assertTrue(versionutils.is_compatible('1.0', '1.0'))
    self.assertTrue(versionutils.is_compatible('1.0.0', '1.0.0'))