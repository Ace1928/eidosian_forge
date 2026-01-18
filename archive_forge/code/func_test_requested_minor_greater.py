from oslotest import base as test_base
from oslo_utils import versionutils
def test_requested_minor_greater(self):
    self.assertFalse(versionutils.is_compatible('1.1', '1.0'))