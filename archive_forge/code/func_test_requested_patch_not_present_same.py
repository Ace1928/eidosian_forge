from oslotest import base as test_base
from oslo_utils import versionutils
def test_requested_patch_not_present_same(self):
    self.assertTrue(versionutils.is_compatible('1.0', '1.0.0'))