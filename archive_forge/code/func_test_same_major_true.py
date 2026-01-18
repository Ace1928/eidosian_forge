from oslotest import base as test_base
from oslo_utils import versionutils
def test_same_major_true(self):
    """Even though the current version is 2.0, since `same_major` defaults
        to `True`, 1.0 is deemed incompatible.
        """
    self.assertFalse(versionutils.is_compatible('2.0', '1.0'))
    self.assertTrue(versionutils.is_compatible('1.0', '1.0'))
    self.assertFalse(versionutils.is_compatible('1.0', '2.0'))